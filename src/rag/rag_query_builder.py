"""
src/rag/rag_query_builder.py
================================================================================
Module : RAG Query Builder — LLM Pre-RAG

Rôle : Avant d'interroger le ChunkFilter (FAISS), ce module utilise le LLM
       Groq pour construire une requête RAG précise et enrichie à partir de :
         - La question brute du parent (déjà traduite en FR)
         - Les 5 derniers messages de la conversation
         - Les mots-clés cliniques extraits (MemoryManager)
         - Le résumé de conversation (MemoryManager)
         - Le profil enfant (trouble, âge, flags comportementaux)

Position dans main_pipeline.py :
  [1] Multimodal
  [2] Langue
  [3] Domain Guard
  [4] Cas critique
  ↓
  [5] RAGQueryBuilder  ← CE MODULE (nouveau)
      Entrée  : question_fr + conversation + profil
      Sortie  : rag_query (str) + rag_tags (list)
  ↓
  [6] ChunkFilter (utilise rag_query au lieu de question brute)
  [7] MemoryManager
  [8] PromptBuilder
  [9] LLM Groq → réponse finale
  [10] Traduction
  [11] PostProcess

Pourquoi un LLM pre-RAG ?
  La question brute du parent est souvent courte, ambiguë ou dépendante du
  contexte ("et pour ça ?" / "comment je fais ?"). Le LLM pre-RAG :
    1. Résout les coréférences avec les 5 derniers messages
    2. Ajoute les termes cliniques du profil (TSA, PECS, TEACCH, etc.)
    3. Reformule en une requête sémantiquement riche pour FAISS
    4. Extrait des tags de filtrage (trouble, domaine, âge)

Exemple :
  Question brute  : "et pour les crises le soir ?"
  Contexte        : conv sur gestion TSA enfant 5 ans non-verbal
  rag_query       : "gestion des crises comportementales TSA enfant non verbal
                     stratégies TEACCH routine nocturne apaisement sensoriel"
  rag_tags        : ["TSA", "crise", "sensoriel", "non_verbal", "4-6"]
================================================================================
"""

import logging
import os

from ..llm.llm_client import LLMClient
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Constantes ────────────────────────────────────────────────────────────────

MAX_TOKENS_PRE_RAG = 200   # court : on veut juste une requête, pas une réponse
TEMPERATURE_PRE_RAG = 0.2  # très bas pour être déterministe et précis

# Tags de troubles pour filtrage ChunkFilter
TROUBLE_TAGS = {
    "tsa"    : ["autisme", "autiste", "tsa", "spectre", "asd", "asperger",
                "pecs", "teacch", "aba", "pictogramme", "non verbal"],
    "rm"     : ["déficience", "retard mental", "di", "intellectuel",
                "autonomie", "apprentissage", "rm"],
    "mixte"  : ["mixte", "comorbidité", "tsa et di", "tsdi"],
    "normal" : ["développement", "normal", "typique"],
}

# Domaines cliniques → pour enrichissement requête
DOMAIN_KEYWORDS = {
    "communication"  : ["communication", "parole", "langage", "verbal", "pecs",
                        "pictogramme", "aac", "echolalie", "muet", "silence"],
    "comportement"   : ["crise", "agression", "automutilation", "colère",
                        "comportement", "morsure", "frappe", "destruction"],
    "sensoriel"      : ["sensoriel", "hypersensibilité", "bruit", "lumière",
                        "toucher", "olfactif", "proprioception"],
    "scolarite"      : ["école", "scolarité", "apprentissage", "inclusion",
                        "classe", "enseignant", "atsem"],
    "alimentation"   : ["manger", "nourriture", "alimentation", "sélectif",
                        "texture", "repas", "boire"],
    "sommeil"        : ["sommeil", "nuit", "réveille", "endormissement",
                        "insomnie", "fatigue"],
    "autonomie"      : ["autonomie", "habillage", "toilettes", "lavage",
                        "propre", "seul", "indépendant"],
    "social"         : ["social", "amis", "jeu", "interaction", "partage",
                        "empathie", "relation"],
}


# ── CLASSE PRINCIPALE ─────────────────────────────────────────────────────────

class RAGQueryBuilder:
    """
    LLM Pre-RAG : construit une requête FAISS enrichie avant ChunkFilter.

    Usage dans main_pipeline.py :
        rag_query_builder = RAGQueryBuilder(llm_client=self.llm)

        # Entre étapes [4] et [5] du pipeline
        rag_result = rag_query_builder.build(
            question    = question_fr,
            conversation = conversation,   # {last_5_messages, keywords, summary}
            profile     = profile,         # profil enfant complet
        )
        rag_query = rag_result["rag_query"]
        rag_tags  = rag_result["rag_tags"]

        # Passer rag_query à ChunkFilter au lieu de question brute
        chunks = self.chunk_filter.search(question=rag_query, profile=profile)
    """

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client : instance LLMClient (groq) depuis main_pipeline.py
                         Si None → fallback par règles (sans LLM)
        """
        self.llm = llm_client
        logger.info("RAGQueryBuilder initialisé ✔")

    @property
    def has_llm(self) -> bool:
        return self.llm is not None

    # ── POINT D'ENTRÉE PRINCIPAL ──────────────────────────────────────────────

    def build(self, question: str,
              conversation: dict = None,
              profile: dict = None) -> dict:
        """
        Construit la requête RAG enrichie.

        Args:
            question     : question du parent (déjà en français)
            conversation : {
                last_5_messages : [{"role": "user"|"assistant", "content": "..."}],
                keywords        : ["crise", "PECS", "non verbal", ...],
                summary         : "résumé glissant de la conversation...",
                total_messages  : 7,
            }
            profile      : profil enfant (dict API)

        Returns:
            {
                "rag_query"      : "gestion crises TSA enfant non verbal TEACCH...",
                "rag_tags"       : ["TSA", "4-6", "comportement", "communication"],
                "original_question": "et pour les crises le soir ?",
                "enriched"       : True,   # True si LLM utilisé
                "method"         : "llm" | "rules",
            }
        """
        conversation = conversation or {}
        profile      = profile      or {}

        # ── 1. Extraire le contexte utile ─────────────────────────────────────
        last_5    = conversation.get("last_5_messages", [])
        keywords  = conversation.get("keywords",        [])
        summary   = conversation.get("summary",         "")
        trouble   = str(profile.get("prediction", "TSA")).upper()

        # ── 2. Détecter le domaine et tags de base par règles ─────────────────
        base_tags = self._extract_tags_rules(question, keywords, trouble, profile)

        # ── 3. Construire la requête ───────────────────────────────────────────
        if self.has_llm and (last_5 or keywords or summary):
            # Mode LLM : reformulation intelligente
            try:
                rag_query = self._build_with_llm(
                    question, last_5, keywords, summary, trouble, profile
                )
                method = "llm"
                logger.info(f"RAGQueryBuilder [LLM] → {len(rag_query)} chars")
            except Exception as e:
                logger.warning(f"LLM pre-RAG échoué : {e} → fallback règles")
                rag_query = self._build_with_rules(question, keywords, trouble, profile)
                method = "rules_fallback"
        else:
            # Mode règles : enrichissement par mots-clés
            rag_query = self._build_with_rules(question, keywords, trouble, profile)
            method = "rules"
            logger.info(f"RAGQueryBuilder [règles] → {len(rag_query)} chars")

        logger.info(f"RAG query ({method}) : {rag_query[:120]}...")

        return {
            "rag_query"        : rag_query,
            "rag_tags"         : base_tags,
            "original_question": question,
            "enriched"         : bool(last_5 or keywords),
            "method"           : method,
        }

    # ── CONSTRUCTION PAR LLM ─────────────────────────────────────────────────

    def _build_with_llm(self, question: str, last_5: list,
                         keywords: list, summary: str,
                         trouble: str, profile: dict) -> str:
        """
        Utilise le LLM Groq pour reformuler la question en requête RAG optimisée.

        Le prompt est court et très directif : on veut une sortie de 1-3 phrases
        maximum, riche en termes cliniques, sans explication.
        """
        logger.info(f'LAST 5 MESSAGES: {last_5}')
        # Construire le contexte conversationnel (5 derniers échanges)
        conv_block = ""
        if last_5:
            conv_lines = []
            for msg in last_5[-5:]:  # max 5
                role    = "Parent" if msg.get("role") == "user" else "Assistant"
                content = (msg.get("content") or "").strip()
                if content:
                    # Tronquer chaque message à 150 chars pour rester concis
                    conv_lines.append(f"{role}: {content[:150]}")
            if conv_lines:
                conv_block = "\n".join(conv_lines)

        # Contexte profil minimal
        age_val = profile.get("Age_Years") or profile.get("age")
        profil_block = f"Enfant : {trouble}"
        if age_val:
            profil_block += f", {age_val} ans"
        if profile.get("PR_QF1A") in (2, 3):
            profil_block += ", non verbal"
        if profile.get("PR_QO1_A_COMBINE") == 1:
            profil_block += ", agression active"
        if profile.get("PR_QO1_C_COMBINE") == 1:
            profil_block += ", automutilation"
        if keywords:
            profil_block += f". Sujets clés : {', '.join(keywords[:8])}"

        # Prompt ultra-court et directif
        prompt = f"""Tu es un expert en TSA et déficience intellectuelle. 
Génère une requête de recherche documentaire en français (2-3 phrases max) 
pour trouver les documents les plus pertinents.

{f"Résumé conversation : {summary[:200]}" if summary else ""}
{f"Contexte récent :{chr(10)}{conv_block}" if conv_block else ""}
Profil : {profil_block}
Question du parent : "{question}"

Requête de recherche (termes cliniques précis, méthodes, comportements) :"""

        # Appel LLM avec tokens très limités
        rag_query = self.llm.generate(
            prompt,
            max_tokens=MAX_TOKENS_PRE_RAG,
            temperature=TEMPERATURE_PRE_RAG,
        )

        # Nettoyer la sortie (supprimer guillemets, préfixes, etc.)
        rag_query = rag_query.strip().strip('"').strip("'")
        rag_query = rag_query.replace("Requête :", "").replace("Requête de recherche :", "").strip()

        # Fallback si la sortie est trop courte ou invalide
        if len(rag_query.split()) < 4:
            logger.warning("Sortie LLM pre-RAG trop courte → fallback règles")
            return self._build_with_rules(question, keywords, trouble, profile)

        return rag_query

    # ── CONSTRUCTION PAR RÈGLES (fallback) ────────────────────────────────────

    def _build_with_rules(self, question: str, keywords: list,
                           trouble: str, profile: dict) -> str:
        """
        Enrichissement par règles : question + mots-clés trouble + flags profil.
        Utilisé quand pas de contexte conversationnel OU si LLM échoue.
        """
        parts = [question.strip()]

        # Ajouter le type de trouble
        trouble_terms = {
            "TSA"   : "autisme TSA trouble spectre autistique",
            "RM"    : "déficience intellectuelle retard mental autonomie",
            "MIXTE" : "autisme déficience intellectuelle TSA DI",
        }
        if trouble in trouble_terms:
            parts.append(trouble_terms[trouble])

        # Ajouter les mots-clés cliniques de la conversation
        if keywords:
            parts.append(" ".join(keywords[:8]))

        # Ajouter les termes liés aux flags actifs du profil
        profile_terms = []
        if profile.get("PR_QF1A") in (2, 3):
            profile_terms.extend(["communication alternative PECS pictogramme non verbal AAC"])
        if profile.get("PR_QO1_A_COMBINE") == 1:
            profile_terms.extend(["gestion agression comportement crise désescalade"])
        if profile.get("PR_QO1_C_COMBINE") == 1:
            profile_terms.extend(["automutilation prévention intervention comportementale"])
        if profile.get("PR_QO1_E_COMBINE") == 1:
            profile_terms.extend(["fugue errance sécurité enfant autiste"])
        if profile.get("PR_QN1_G") in (1, 2):
            profile_terms.extend(["épilepsie crises convulsions enfant"])
        if profile.get("PR_QK1", 1) >= 4:
            profile_terms.extend(["alimentation sélective troubles alimentaires enfant autiste"])
        if profile.get("PR_QQ", 1) >= 3:
            profile_terms.extend(["accompagnement intensif soutien parental"])

        if profile_terms:
            parts.append(" ".join(profile_terms))

        # Ajouter termes méthodologiques standard TSA
        if trouble in ("TSA", "MIXTE"):
            parts.append("TEACCH ABA méthode structure routine apprentissage")

        return " ".join(parts)

    # ── EXTRACTION TAGS ───────────────────────────────────────────────────────

    def _extract_tags_rules(self, question: str, keywords: list,
                             trouble: str, profile: dict) -> list:
        """
        Extrait les tags de filtrage pour ChunkFilter :
          - trouble : TSA / RM / MIXTE / Normal
          - age_group : 0-3 / 4-6 / 7-12 / 13+
          - domaines cliniques détectés
        """
        tags = []

        # Tag trouble
        if trouble in ("TSA", "RM", "MIXTE", "NORMAL"):
            tags.append(trouble)

        age= profile.get("Age_Years")

        if age is not None:
            try:
                age = int(age)
                if   age <= 3:  tags.append("0-3")
                elif age <= 6:  tags.append("4-6")
                elif age <= 12: tags.append("7-12")
                else:           tags.append("13+")
            except (ValueError, TypeError):
                pass

        # Tags domaines depuis question + mots-clés
        text = f"{question} {' '.join(keywords)}".lower()
        for domain, kws in DOMAIN_KEYWORDS.items():
            if any(kw in text for kw in kws):
                tags.append(domain)

        # Tags flags comportementaux actifs
        def val(k): return profile.get(k)
        if val("PR_QO1_A_COMBINE") == 1: tags.append("agression")
        if val("PR_QO1_C_COMBINE") == 1: tags.append("automutilation")
        if val("PR_QO1_E_COMBINE") == 1: tags.append("fugue")
        if val("PR_QF1A") in (2, 3):    tags.append("non_verbal")
        if val("PR_QN1_G") in (1, 2):   tags.append("epilepsie")

        return list(dict.fromkeys(tags))  # dédupliquer en conservant l'ordre


# ─────────────────────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*70)
    print("  TEST — RAGQueryBuilder (mode règles sans LLM)")
    print("="*70)

    from dotenv import load_dotenv
    load_dotenv()
    llm     = LLMClient(api_key=os.getenv("GROQ_API_KEY"))

    builder = RAGQueryBuilder(llm_client=llm)

    # Cas 1 : question courte avec contexte
    result = builder.build(
        question = "et pour les crises le soir ?",
        conversation = {
            "last_5_messages": [
                {"role": "user",      "content": "Mon fils fait des crises le matin."},
                {"role": "assistant", "content": "Les crises matinales sont souvent liées aux transitions."},
                {"role": "user",      "content": "Il est aussi non verbal depuis ses 3 ans."},
                {"role": "assistant", "content": "Le PECS est adapté pour les enfants non verbaux."},
                {"role": "user",      "content": "et pour les crises le soir ?"},
            ],
            "keywords"  : ["crise", "non verbal", "PECS", "transition"],
            "summary"   : "Enfant TSA 5 ans non verbal, crises fréquentes matin et soir.",
            "total_messages": 8,
        },
        profile = {
            "prediction"      : "TSA",
            "Age_Years"       : 5,
            "Sex"             : "M",
            "PR_QF1A"         : 3,   # non verbal
            "PR_QO1_A_COMBINE": 1,   # agression active
            "PR_QQ"           : 3,   # soutien modéré
        }
    )

    print(f"\n  Question originale : '{result['original_question']}'")
    print(f"  Méthode           : {result['method']}")
    print(f"  Enrichie          : {result['enriched']}")
    print(f"  Tags RAG          : {result['rag_tags']}")
    print(f"\n  Requête RAG :")
    print(f"  {result['rag_query']}")

    # Cas 2 : question directe sans contexte
    result2 = builder.build(
        question = "Comment gérer l'automutilation de mon enfant ?",
        conversation = {"last_5_messages": [], "keywords": [], "summary": ""},
        profile = {
            "prediction"      : "RM",
            "Age_Years"       : 8,
            "PR_QO1_C_COMBINE": 1,
            "PR_QQ"           : 4,
        }
    )
    print(f"\n  Question #2 : '{result2['original_question']}'")
    print(f"  Tags RAG    : {result2['rag_tags']}")
    print(f"  Requête RAG : {result2['rag_query'][:150]}...")
    print("="*70)
