import os, logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from groq import Groq
    GROQ_OK = True
except ImportError:
    GROQ_OK = False
    logger.warning("groq non installé → pip install groq")


# CONFIGURATION

GROQ_MODEL       = "llama-3.3-70b-versatile"   # meilleur modèle Groq gratuit
MAX_TOKENS       = 1024
TEMPERATURE      = 0.4    # bas = réponses cohérentes et factuelles
TOP_P            = 0.9

# Nombre de questions suivantes à suggérer
N_SUGGESTED_QUESTIONS = 3


# PROMPT BUILDER

class PromptBuilder:
    """
    Construit le prompt final structuré pour le LLM.

    Assemblage :
        [1] Système        : rôle + règles strictes
        [2] Profil enfant  : contexte depuis profile_filter
        [3] Profil détecté : remarques accumulées depuis les conversations
        [4] Mémoire        : résumé + mots-clés + 5 derniers messages
        [5] Sources RAG    : chunks réponse + chunks profil
        [6] Question       : message du parent
        [7] Instructions   : structure de la réponse attendue
    """

    # ── PROMPT SYSTÈME ────────────────────────────────────────────────────────

    SYSTEM_PROMPT = """Tu es un ami de confiance qui accompagne les parents d'enfants autistes ou avec une déficience intellectuelle. Tu parles comme un ami proche qui connait très bien le sujet — pas comme un médecin ou un rapport officiel.

Ton style :
- Chaleureux, direct, humain — comme un ami qui comprend vraiment
- Pas de jargon médical inutile, pas de formules pompeux
- Des conseils concrets que le parent peut appliquer aujourd'hui
- Court et clair — pas de blabla inutile

Tes limites (non négociables) :
- Tu ne poses jamais de diagnostic médical
- Tu ne prescris jamais de médicaments
- Si la situation est grave (automutilation, violence sévère), tu orientes vers un professionnel
- Tes conseils s'appuient sur les sources scientifiques fournies"""

    # ── CONSTRUCTION DU PROMPT ────────────────────────────────────────────────

    def build(self,
              question         : str,
              profile_context  : str  = "",
              profile_detecter : list = None,
              memory_block     : str  = "",
              rag_block        : str  = "",
              parent_lang      : str  = "fr",
              media_description: str  = "",
              media_type       : str  = "") -> list:
        """
        Construit le prompt complet sous forme de messages Groq.

        Args:
            question          : question du parent (en français)
            profile_context   : bloc profil depuis profile_filter.py
            profile_detecter  : remarques accumulées depuis les conversations
            memory_block      : bloc mémoire depuis memory_manager.py
            rag_block         : bloc RAG depuis chunk_filter.py
            parent_lang       : langue du parent pour la réponse finale
            media_description : description extraite d'une image ou vidéo
            media_type        : "image" | "video" | "" (si pas de media)

        Returns:
            Liste de messages [{"role": ..., "content": ...}]
        """
        profile_detecter = profile_detecter or []

        user_content = self._build_user_content(
            question, profile_context, profile_detecter,
            memory_block, rag_block, parent_lang,
            media_description, media_type
        )

        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": user_content},
        ]

    def _build_user_content(self, question, profile_context,
                             profile_detecter, memory_block,
                             rag_block, parent_lang,
                             media_description="", media_type="") -> str:
        sections = []

        # ── [1] Profil de l'enfant ─────────────────────────────────────
        if profile_context and profile_context.strip():
            sections.append(
                "━━━ PROFIL DE L'ENFANT ━━━\n"
                + profile_context.strip()
            )

        # ── [2] Observations accumulées ───────────────────────────────
        if profile_detecter:
            obs = "\n".join(f"• {r}" for r in profile_detecter)
            sections.append(
                "━━━ OBSERVATIONS ISSUES DES CONVERSATIONS ━━━\n" + obs
            )

        # ── [3] Mémoire conversationnelle ──────────────────────────────
        if memory_block and memory_block.strip():
            sections.append(
                "━━━ HISTORIQUE DE LA CONVERSATION ━━━\n"
                + memory_block.strip()
            )

        # ── [4] Description média (image ou vidéo) ─────────────────────
        if media_description and media_description.strip():
            media_label = {
                "image": "IMAGE ENVOYÉE PAR LE PARENT",
                "video": "VIDÉO ENVOYÉE PAR LE PARENT",
            }.get(media_type.lower(), "MÉDIA ENVOYÉ PAR LE PARENT")

            sections.append(
                f"━━━ {media_label} ━━━\n"
                f"Description extraite automatiquement :\n"
                f"{media_description.strip()}"
            )

        # ── [5] Sources scientifiques (RAG) ───────────────────────────
        if rag_block and rag_block.strip():
            sections.append(
                "━━━ SOURCES SCIENTIFIQUES ━━━\n"
                + rag_block.strip()
            )

        # ── [6] Question du parent ─────────────────────────────────────
        sections.append(
            "━━━ QUESTION DU PARENT ━━━\n" + question.strip()
        )

        # ── [7] Instructions de réponse ────────────────────────────────
        lang_instruction = (
            f"\nIMPORTANT : Le parent parle {self._lang_label(parent_lang)}. "
            f"Réponds en {self._lang_label(parent_lang)}."
            if parent_lang != "fr" else ""
        )

        media_instruction = (
            "\n- Si pertinent, tiens compte du contenu du média envoyé "
            "par le parent dans ta réponse."
            if media_description else ""
        )

        instructions = (
            "━━━ INSTRUCTIONS ━━━\n"
            "Réponds comme un ami proche — naturel, chaleureux, direct.\n\n"
            "Structure :\n\n"
            "**[Compréhension]**\n"
            "1-2 phrases max. Montre que tu comprends vraiment ce que vit le parent "
            "(pas une reformulation froide — une vraie réaction humaine).\n\n"
            "**[Conseils]**\n"
            "- Donne des conseils concrets, applicables aujourd'hui\n"
            "- Chaque conseil : 2-3 phrases claires, simples, directes\n"
            "- Après chaque conseil, cite la source naturellement : "
            "*(d'après [nom source])*\n"
            "- Si question simple : 2-3 conseils\n"
            "- Si question complexe : 4-5 conseils\n"
            "- Pas de titres de section, pas de bullet points formels — "
            "juste des paragraphes qui s'enchaînent naturellement\n\n"
            "**[Questions pour aller plus loin]**\n"
            f"Propose {N_SUGGESTED_QUESTIONS} questions courtes et concrètes "
            "pour continuer la conversation. Format simple : liste numérotée."
            + media_instruction
            + lang_instruction
        )
        sections.append(instructions)

        return "\n\n".join(sections)

    def _lang_label(self, code: str) -> str:
        return {"fr": "français", "ar": "arabe", "en": "anglais"}.get(code, "français")


# CLIENT LLM

class LLMClient:
    """
    Client Groq pour générer des réponses via Llama 3.3 70B.
    """

    def __init__(self, api_key: str = None):
        """
        Args:
            api_key : clé Groq (ou variable d'env GROQ_API_KEY)
        """
        if not GROQ_OK:
            raise ImportError("pip install groq")

        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "Clé Groq manquante.\n"
                "  1. Créer un compte sur https://console.groq.com\n"
                "  2. Générer une API Key\n"
                "  3. Ajouter dans .env : GROQ_API_KEY=gsk_..."
            )

        self.client = Groq(api_key=key)
        self.model  = GROQ_MODEL
        logger.info(f"LLMClient initialisé ✔ (modèle : {self.model})")

    # ── GÉNÉRATION TEXTE ──────────────────────────────────────────────────────

    def generate(self, prompt: str,
                 max_tokens : int   = MAX_TOKENS,
                 temperature: float = TEMPERATURE) -> str:
        """
        Génère une réponse depuis un prompt texte simple.
        Utilisé par memory_manager.py, profile_updater.py, video_handler.py

        Args:
            prompt : texte du prompt

        Returns:
            Réponse du LLM (str)
        """
        messages = [{"role": "user", "content": prompt}]
        return self._call(messages, max_tokens, temperature)

    # ── GÉNÉRATION DEPUIS MESSAGES ────────────────────────────────────────────

    def generate_from_messages(self, messages: list,
                                max_tokens : int   = MAX_TOKENS,
                                temperature: float = TEMPERATURE) -> str:
        """
        Génère une réponse depuis une liste de messages structurés.
        Utilisé par le pipeline principal avec PromptBuilder.

        Args:
            messages : [{"role": "system"|"user", "content": "..."}]

        Returns:
            Réponse du LLM (str)
        """
        return self._call(messages, max_tokens, temperature)

    # ── APPEL API ─────────────────────────────────────────────────────────────

    def _call(self, messages: list, max_tokens: int,
              temperature: float) -> str:
        """Appel Groq API avec gestion d'erreurs."""
        try:
            response = self.client.chat.completions.create(
                model       = self.model,
                messages    = messages,
                max_tokens  = max_tokens,
                temperature = temperature,
                top_p       = TOP_P,
            )
            text = response.choices[0].message.content.strip()
            logger.info(
                f"LLM ✔ | tokens={response.usage.total_tokens} | "
                f"chars={len(text)}"
            )
            return text

        except Exception as e:
            logger.error(f"Erreur Groq API : {e}")
            raise RuntimeError(f"Erreur LLM : {e}")


# TEST

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # ── Test PromptBuilder ────────────────────────────────────────────────
    builder = PromptBuilder()

    messages = builder.build(
        question = "Comment gérer les crises de mon enfant le nuit ?",

        profile_context = (
            "Profil de l'enfant : garçon de 5 ans.\n"
            "Diagnostic estimé : Trouble du Spectre Autistique (TSA) (confiance : 89%).\n"
            "Niveau de soutien requis : Soutien quotidien modéré (niveau 3/4).\n"
            "Communication : Pas de communication verbale.\n"
            "→ Enfant non verbal : recommander des outils AAC (PECS, pictogrammes).\n"
            "Comportements difficiles actifs : agression physique, automutilation.\n"
            "→ Priorité : stratégies de gestion de crise.\n"
            "Modules de connaissances activés : tsa, teacch, aac, crise, sensoriel."
        ),

        profile_detecter = [
            "Non verbal / pas de langage oral",
            "Crises fréquentes le matin",
            "Hypersensibilité au bruit",
            "Refuse certains aliments",
        ],

        memory_block = (
            "=== Résumé de la conversation ===\n"
            "Parent inquiet pour les crises matinales de son fils TSA de 5 ans.\n"
            "A déjà essayé les pictogrammes avec peu de succès.\n\n"
            "=== Sujets abordés ===\n"
            "crise, matin, pictogramme, PECS\n\n"
            "=== Derniers échanges ===\n"
            "Parent    : Il crie dès qu'on le réveille.\n"
            "Assistant : Les transitions sont difficiles pour les enfants TSA."
        ),

        rag_block = (
            "=== Sources de réponse ===\n"
            "[1] (score=0.82) (Autisme Info Service) "
            "L'agressivité matinale chez l'enfant autiste est souvent liée "
            "aux transitions brusques. Préparer l'enfant avec des emplois du "
            "temps visuels réduit significativement ces comportements...\n\n"
            "=== Contexte profil ===\n"
            "[Communication]\n"
            "[1] (score=0.79) (Maison de l'Autisme) "
            "Le PECS phase 1 consiste à échanger une image contre un objet désiré. "
            "Cette méthode est efficace dès 18 mois pour les enfants non verbaux..."
        ),
    )

    print("\n" + "="*60)
    print("  PROMPT GÉNÉRÉ")
    print("="*60)
    print(f"  Messages : {len(messages)}")
    print(f"\n  [SYSTEM] :\n{messages[0]['content'][:200]}...")
    print(f"\n  [USER] (extrait) :\n{messages[1]['content'][:500]}...")

    # ── Test LLM (si clé dispo) ───────────────────────────────────────────
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        print("\n" + "="*60)
        print("  TEST LLM — Groq")
        print("="*60)
        try:
            client   = LLMClient(api_key=api_key)
            response = client.generate_from_messages(messages)
            print(f"\n{response}")
        except Exception as e:
            print(f"  ✘ Erreur : {e}")
    else:
        print("\n  ⚠️  GROQ_API_KEY non défini dans .env → test LLM ignoré")
        print("  Ajouter : GROQ_API_KEY=gsk_... dans votre fichier .env")

    print("\n" + "="*60)