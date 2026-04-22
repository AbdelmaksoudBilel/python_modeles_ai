"""
src/llm/main_pipeline.py — version avec LLM Pre-RAG
================================================================================
Pipeline complet :

  question + profil + conversation + child + media(optionnel)
    │
    ├─[1]  Multimodal      → image/vidéo/audio → texte FR
    ├─[2]  Langue          → détection + traduction FR
    ├─[3]  Domain Guard    → hors domaine → réponse polie
    ├─[4]  Cas critique    → danger → orienter urgences
    │
    ├─[5]  RAGQueryBuilder ← NOUVEAU : LLM pre-RAG
    │       Entrée  : question_fr + last_5_messages + keywords + summary + profil
    │       Sortie  : rag_query (str enrichie) + rag_tags (list)
    │       But     : résoudre coréférences, ajouter termes cliniques,
    │                 construire requête FAISS sémantiquement riche
    │
    ├─[6]  ChunkFilter     → double RAG (rag_query) + scoring + web search
    ├─[7]  MemoryManager   → résumé + mots-clés + 5 msgs
    ├─[8]  PromptBuilder   → assemblage prompt final
    ├─[9]  LLM Groq        → génération réponse FR
    ├─[10] Traduction       → réponse langue parent
    └─[11] PostProcess     → update mémoire + profil → DB

Différence clé vs ancienne version :
  Avant : ChunkFilter recevait `question_fr` brute (courte, ambiguë)
  Après : ChunkFilter reçoit `rag_query` enrichie par LLM pre-RAG
          → meilleure précision FAISS, moins de web search déclenché
================================================================================
"""

import os, logging
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Imports modules ───────────────────────────────────────────────────────────
from ..multimodal.language_handler import LanguageHandler
from ..multimodal.image_handler    import ImageHandler
from ..multimodal.video_handler    import VideoHandler
from ..rag.rag_query_builder import RAGQueryBuilder   # ← NOUVEAU
from ..rag.chunk_filter      import ChunkFilter
from ..rag.memory_manager    import MemoryManager
from ..rag.profile_updater   import ProfileUpdater
from .llm_client             import LLMClient, PromptBuilder


# ── Configuration ─────────────────────────────────────────────────────────────

CHUNKS_FILE = "data/rag/chunk/rag_chunks_meta.json"
FAISS_FILE  = "data/rag/vector/faiss_index.bin"
META_FILE   = "data/rag/vector/metadata.json"

CRITICAL_KEYWORDS = [
    "suicide", "se tuer", "mourir", "mort", "danger de mort",
    "urgence", "hôpital", "crise grave", "convulsion",
]

DOMAIN_KEYWORDS = [
    # ───────────── Général ─────────────
    "autisme", "tsa", "trouble spectre autistique", "asperger",
    "déficience intellectuelle", "retard mental", "handicap mental",
    "trouble neurodéveloppemental", "développement",

    # ───────────── Enfant / famille ─────────────
    "enfant", "bébé", "ado", "adolescent",
    "fils", "fille", "mon enfant", "mon fils", "ma fille",
    "parent", "famille",

    # ───────────── Comportement ─────────────
    "comportement", "crise", "colère", "agression", "violence",
    "automutilation", "frapper", "mordre", "hurler",
    "pleurer", "tantrum", "stéréotypie", "répétition",
    "hyperactivité", "impulsivité",

    # ───────────── Communication ─────────────
    "communication", "langage", "parler", "verbal", "non verbal",
    "retard langage", "écholalie", "compréhension",
    "expression", "gestes", "pictogramme",
    "pecs", "aac",

    # ───────────── Social ─────────────
    "interaction", "social", "regard", "contact visuel",
    "isolement", "amis", "relation",

    # ───────────── Sensoriel ─────────────
    "sensoriel", "hypersensibilité", "hyposensibilité",
    "bruit", "lumière", "toucher", "texture",
    "sensibilité", "stimuli",

    # ───────────── Apprentissage ─────────────
    "apprentissage", "école", "classe", "enseignant",
    "concentration", "attention", "mémoire",
    "éducation", "pédagogie",

    # ───────────── Santé / quotidien ─────────────
    "sommeil", "dormir", "insomnie",
    "alimentation", "manger", "refus alimentaire",
    "toilette", "propreté", "habillage",
    "routine", "changement", "transition",

    # ───────────── Émotion ─────────────
    "anxiété", "stress", "peur", "émotion",
    "frustration", "angoisse",

    # ───────────── Diagnostic ─────────────
    "diagnostic", "test", "évaluation",
    "symptôme", "signes", "dépistage",

    # ───────────── Thérapies ─────────────
    "thérapie", "traitement", "prise en charge",
    "aba", "teacch", "orthophonie",
    "ergothérapie", "psychologue", "psychiatre",
    "intervention", "rééducation",

    # ───────────── Autonomie ─────────────
    "autonomie", "indépendance",
    "apprendre", "habitude",
]


# ── PIPELINE PRINCIPAL ────────────────────────────────────────────────────────

class MainPipeline:

    def __init__(self):
        logger.info("Initialisation du pipeline...")

        # LLM principal (Groq llama-3.3-70b)
        self.llm     = LLMClient(api_key=os.getenv("GROQ_API_KEY"))
        self.builder = PromptBuilder()

        # Langue
        self.lang_handler = LanguageHandler()

        # Multimodal
        self.image_handler  = ImageHandler(llm_client=None)
        self.video_handler  = VideoHandler()
        # SpeechHandler removed — leave None as fallback
        self.speech_handler = None

        # RAG
        self.chunk_filter = ChunkFilter(CHUNKS_FILE, FAISS_FILE, META_FILE)

        # ── NOUVEAU : LLM Pre-RAG Query Builder ───────────────────────────────
        # Utilise le même LLM Groq mais avec max_tokens=200 et temperature=0.2
        # pour construire une requête FAISS enrichie avant ChunkFilter
        self.rag_query_builder = RAGQueryBuilder(llm_client=self.llm)

        # Mémoire + Profil
        self.memory_manager  = MemoryManager(llm_client=self.llm)
        self.profile_updater = ProfileUpdater(llm_client=self.llm)

        logger.info("Pipeline prêt ✔ (avec LLM pre-RAG)")

    # ── PIPELINE PRINCIPAL ────────────────────────────────────────────────────

    def run(self,
            question    : str,
            profile     : dict,
            conversation: dict = None,
            child       : dict = None,
            media_path  : str  = "",
            media_type  : str  = "") -> dict:
        """
        Exécute le pipeline complet.

        Args:
            question     : message du parent
            profile      : profil enfant depuis la DB (JSON API)
            conversation : {
                last_5_messages : [{"role":"user"|"assistant","content":"..."}],
                summary         : "résumé glissant...",
                keywords        : ["crise","PECS",...],
                total_messages  : 7,
            }
            child        : { id, profile_detecter }
            media_path   : chemin image/vidéo/audio (optionnel)
            media_type   : "image"|"video"|"audio"|""

        Returns:
            {
                "answer"          : réponse finale (langue parent),
                "answer_fr"       : réponse en français (interne),
                "parent_lang"     : langue détectée,
                "rag_score"       : score RAG moyen,
                "web_triggered"   : bool,
                "domain_blocked"  : bool,
                "critical_alert"  : bool,
                "rag_query"       : requête enrichie envoyée à FAISS,  ← NOUVEAU
                "rag_query_method": "llm"|"rules"|"rules_fallback",    ← NOUVEAU
                "updates"         : {
                    "summary"         : nouveau résumé,
                    "keywords"        : nouveaux mots-clés,
                    "profile_detecter": profil mis à jour,
                    "should_update_db": bool,
                }
            }
        """
        conversation = conversation or {}
        child        = child        or {}
        result       = self._empty_result()

        # ── [1] Traitement multimodal ─────────────────────────────────────────
        media_text, media_description = self._process_media(media_path, media_type)

        # ── [2] Détection langue + traduction FR ──────────────────────────────
        lang_result = self.lang_handler.process(question)
        parent_lang = lang_result["detected_lang"]
        question_fr = lang_result["translated_text"]
        full_input  = f"{question_fr} (media description(image ou vidéo): {media_text})".strip()

        result["parent_lang"] = parent_lang
        print(f"[2] Langue détectée : {parent_lang} | question FR : {question_fr[:60]}...")
        logger.info(f"[2] Langue : {parent_lang} | question_fr={question_fr[:60]}...")


        # ── [4] Alerte cas critique ───────────────────────────────────────────
        if self._is_critical(question_fr):
            logger.warning("[4] Cas critique détecté")
            result["critical_alert"] = True
            result["answer"] = self._critical_response(parent_lang)
            return result

        # ── [5] RAGQueryBuilder — LLM Pre-RAG ────────────────────────────────
        # Construit une requête FAISS enrichie AVANT d'interroger ChunkFilter
        # Utilise : question_fr + last_5_messages + keywords + summary + profil
        logger.info("[5] RAGQueryBuilder : construction requête RAG enrichie...")

        rag_build_result = self.rag_query_builder.build(
            question     = question_fr,
            conversation = conversation,   # contient last_5_messages + keywords + summary
            profile      = profile,        # profil enfant pour extraire flags
        )

        # La requête enrichie remplace question_fr pour le FAISS
        rag_query  = rag_build_result["rag_query"]
        rag_tags   = rag_build_result["rag_tags"]
        rag_method = rag_build_result["method"]

        result["rag_query"]        = rag_query
        result["rag_query_method"] = rag_method

        logger.info(
            f"[5] RAG query ({rag_method}) : {rag_query[:100]}..."
            f" | tags={rag_tags}"
        )

        # ── [3] Domain Guard ──────────────────────────────────────────────────
        if not self._is_in_domain(rag_query):
            logger.warning("[3] Domain Guard : hors domaine")
            result["domain_blocked"] = True
            result["answer"] = self._out_of_domain_response(parent_lang)
            return result
        
        # ── [6] ChunkFilter — Double RAG + scoring ────────────────────────────
        # On passe rag_query (enrichie) au lieu de question_fr (brute)
        logger.info("[6] ChunkFilter : recherche FAISS avec requête enrichie...")

        rag_results = self.chunk_filter.search(
            question = rag_query,   # ← requête enrichie par LLM pre-RAG
            profile  = profile,
        )

        result["rag_score"]    = rag_results["avg_score"]
        result["web_triggered"] = rag_results["web_triggered"]

        logger.info(
            f"[6] RAG score={rag_results['avg_score']:.3f} | "
            f"web={rag_results['web_triggered']} | "
            f"chunks={len(rag_results.get('response_chunks', []))}"
        )

        # ── [7] MemoryManager — résumé + mots-clés + 5 msgs ──────────────────
        logger.info("[7] MemoryManager : bloc mémoire...")
        memory_block = self.memory_manager.build_memory_block(
            last_5_messages = conversation.get("last_5_messages", []),
            summary         = conversation.get("summary",         ""),
            keywords        = conversation.get("keywords",        []),
        )

        # ── [8] PromptBuilder — assemblage prompt final ───────────────────────
        logger.info("[8] PromptBuilder : assemblage prompt...")
        messages = self.builder.build(
            question          = question_fr,
            profile_context   = rag_results["profile_context"],
            profile_detecter  = child.get("profile_detecter", []),
            memory_block      = memory_block,
            rag_block         = rag_results["prompt_block"],
            parent_lang       = parent_lang,
            media_description = media_description,
            media_type        = media_type,
        )

        # ── [9] LLM Groq — génération réponse FR ─────────────────────────────
        logger.info("[9] LLM Groq : génération réponse...")
        answer_fr = self.llm.generate_from_messages(messages)
        result["answer_fr"] = answer_fr

        # ── [10] Traduction réponse → langue parent ───────────────────────────
        if parent_lang != "fr":
            logger.info(f"[10] Traduction réponse → {parent_lang}")
            answer = self.lang_handler.translate_response_to_parent(answer_fr, parent_lang)
        else:
            answer = answer_fr
        result["answer"] = answer

        # ── [11] PostProcess — mise à jour mémoire + profil ───────────────────
        logger.info("[11] PostProcess : update mémoire + profil...")
        updates = self._post_process(
            conversation = conversation,
            child        = child,
            question_fr  = question_fr,
            answer_fr    = answer_fr,
        )
        result["updates"] = updates

        logger.info("Pipeline terminé ✔")
        return result

    # ── MODULES INTERNES ──────────────────────────────────────────────────────

    def _process_media(self, media_path: str, media_type: str):
        if not media_path or not os.path.exists(media_path):
            return "", ""
        try:
            if media_type == "image":
                r = self.image_handler.process(media_path)
                return r.get("translated_text", ""), r.get("translated_text", "")
            elif media_type == "video":
                r = self.video_handler.process(media_path)
                return r.get("translated_text", ""), r.get("translated_text", "")
            elif media_type == "audio":
                # SpeechHandler removed: skip audio processing if not available
                if self.speech_handler:
                    r = self.speech_handler.process(media_path)
                    return r.get("translated_text", ""), ""
                else:
                    logger.info("Audio processing disabled (SpeechHandler supprimé)")
                    return "", ""
        except Exception as e:
            logger.warning(f"Erreur traitement média : {e}")
        return "", ""

    def _is_in_domain(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in DOMAIN_KEYWORDS)

    def _is_critical(self, text: str) -> bool:
        text_lower = text.lower()
        return any(kw in text_lower for kw in CRITICAL_KEYWORDS)

    def _post_process(self, conversation: dict, child: dict,
                      question_fr: str, answer_fr: str) -> dict:
        total_messages = conversation.get("total_messages", 0) + 2

        mem_update = self.memory_manager.update_after_response(
            last_5_messages = conversation.get("last_5_messages", []),
            summary         = conversation.get("summary",         ""),
            keywords        = conversation.get("keywords",        []),
            new_question    = question_fr,
            new_answer      = answer_fr,
            total_messages  = total_messages,
        )

        prof_update = self.profile_updater.update(
            profile_detecter = child.get("profile_detecter", []),
            new_question     = question_fr,
            new_answer       = answer_fr,
        )
        
        return {
            "summary"         : mem_update["summary"],
            "keywords"        : mem_update["keywords"],
            "profile_detecter": prof_update["profile_detecter"],
            "should_update_db": mem_update["should_update_db"] or prof_update["updated"],
            "memory_changes"  : mem_update,
            "profile_changes" : prof_update["changes"],
        }

    def _empty_result(self) -> dict:
        return {
            "answer"           : "",
            "answer_fr"        : "",
            "parent_lang"      : "fr",
            "rag_score"        : 0.0,
            "rag_query"        : "",
            "rag_query_method" : "none",
            "web_triggered"    : False,
            "domain_blocked"   : False,
            "critical_alert"   : False,
            "updates"          : {
                "summary"         : "",
                "keywords"        : [],
                "profile_detecter": [],
                "should_update_db": False,
            },
        }

    def _out_of_domain_response(self, lang: str) -> str:
        msgs = {
            "fr": "Je suis spécialisé dans l'accompagnement des parents d'enfants avec TSA ou déficience intellectuelle. Pourriez-vous reformuler votre question dans ce contexte ?",
            "ar": "أنا متخصص في دعم أولياء أمور الأطفال المصابين بالتوحد أو الإعاقة الذهنية. هل يمكنك إعادة صياغة سؤالك ؟",
            "en": "I specialize in supporting parents of children with ASD or intellectual disabilities. Could you rephrase your question in this context?",
        }
        return msgs.get(lang, msgs["fr"])

    def _critical_response(self, lang: str) -> str:
        msgs = {
            "fr": "⚠️ La situation que vous décrivez nécessite une aide professionnelle immédiate. Contactez votre médecin, un service d'urgence ou appelez le 15 (SAMU).",
            "ar": "⚠️ الوضع الذي تصفه يتطلب مساعدة متخصصة فورية. يرجى الاتصال بطبيبك أو خدمات الطوارئ.",
            "en": "⚠️ The situation you describe requires immediate professional help. Please contact your doctor or emergency services.",
        }
        return msgs.get(lang, msgs["fr"])


# ── TEST ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = MainPipeline()

    result = pipeline.run(
        question = "et pour les crises le soir, comment je fais ?",

        profile = {
            "prediction"         : "TSA",
            "confidence"         : 0.89,
            "Age_Years"          : 5,
            "Sex"                : "M",
            "PR_QF1A"            : 3,    # non verbal
            "PR_QQ"              : 3,    # soutien modéré
            "PR_QN1_D"           : 1,    # anxiété
            "PR_QO1_A_COMBINE"   : 1,    # agression active
        },

        conversation = {
            "last_5_messages": [
                {"role": "user",      "content": "Mon fils fait des crises le matin."},
                {"role": "assistant", "content": "Les transitions sont difficiles pour les enfants TSA."},
                {"role": "user",      "content": "Il est non verbal depuis 3 ans."},
                {"role": "assistant", "content": "Le PECS est adapté pour les enfants non verbaux."},
                {"role": "user",      "content": "et pour les crises le soir, comment je fais ?"},
            ],
            "summary"       : "Enfant TSA 5 ans non verbal, crises fréquentes.",
            "keywords"      : ["TSA", "crise", "non verbal", "PECS"],
            "total_messages": 8,
        },

        child = {
            "id"              : "child_123",
            "profile_detecter": ["non verbal", "crises le matin", "hypersensibilité sonore"],
        },
    )

    print("\n" + "="*70)
    print("  RÉSULTAT PIPELINE (avec LLM pre-RAG)")
    print("="*70)
    print(f"  [5] RAG query ({result['rag_query_method']}) :")
    print(f"      {result['rag_query'][:200]}")
    print(f"  [6] Score RAG     : {result['rag_score']}")
    print(f"  [6] Web déclenché : {result['web_triggered']}")
    print(f"  Langue parent     : {result['parent_lang']}")
    print(f"\n  RÉPONSE :\n{result['answer'][:300]}...")
    print("="*70)