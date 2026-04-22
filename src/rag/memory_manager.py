import re, logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# CONFIGURATION

# Nombre de messages récents à garder dans le prompt
N_LAST_MESSAGES = 5

# Seuil : en dessous → pas de résumé généré
MIN_MESSAGES_FOR_SUMMARY = 5

# Nombre max de mots-clés à conserver
MAX_KEYWORDS = 15

# Mots-clés cliniques à détecter automatiquement (fallback sans LLM)
CLINICAL_KEYWORDS = [
    # Comportements
    "crise", "agression", "automutilation", "fugue", "destruction",
    "colère", "violence", "morsure", "automutilation",
    # Communication
    "verbal", "non-verbal", "PECS", "pictogramme", "langage", "parole",
    "communication", "silence", "AAC",
    # Troubles
    "TSA", "autisme", "autiste", "RM", "déficience", "retard", "TDAH",
    "anxiété", "dépression", "épilepsie", "hypersensibilité",
    # Éducation
    "TEACCH", "ABA", "école", "apprentissage", "autonomie", "routine",
    "structure", "emploi du temps", "pictogramme",
    # Santé
    "sommeil", "alimentation", "médicament", "médecin", "thérapie",
    "douleur", "sensibilité", "sensoriel",
    # Famille
    "fatigue", "épuisement", "aide", "soutien", "inquiet", "peur",
]


# CLASSE PRINCIPALE

class MemoryManager:
    """
    Gère la mémoire conversationnelle pour le prompt LLM.

    Couches :
        - last_5_messages : messages bruts récents
        - summary         : résumé glissant de la conversation
        - keywords        : mots-clés cliniques extraits
    """

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client : client LLM pour générer résumé + mots-clés
                         Si None → extraction par règles (fallback)
        """
        self._llm_client = llm_client
        logger.info("MemoryManager initialisé ✔")

    def set_llm(self, llm_client) -> None:
        """Injecte le LLM après initialisation."""
        self._llm_client = llm_client
        logger.info("LLM défini dans MemoryManager ✔")

    @property
    def has_llm(self) -> bool:
        return self._llm_client is not None

    # ── BLOC MÉMOIRE POUR LE PROMPT ───────────────────────────────────────────

    def build_memory_block(self, last_5_messages: list,
                           summary: str = "",
                           keywords: list = None) -> str:
        """
        Construit le bloc mémoire à injecter dans le prompt LLM.

        Args:
            last_5_messages : liste des 5 derniers messages
                              [{"role": "user"|"assistant", "content": "..."}]
            summary         : résumé glissant depuis la DB
            keywords        : mots-clés depuis la DB

        Returns:
            Bloc texte formaté pour le prompt
        """
        keywords = keywords or []
        lines    = []

        # ── Résumé (si disponible) ────────────────────────────────────────
        if summary and summary.strip():
            lines.append("=== Résumé de la conversation ===")
            lines.append(summary.strip())

        # ── Mots-clés (si disponibles) ────────────────────────────────────
        if keywords:
            kw_str = ", ".join(keywords[:MAX_KEYWORDS])
            lines.append(f"\n=== Sujets abordés ===")
            lines.append(kw_str)

        # ── 5 derniers messages ───────────────────────────────────────────
        if last_5_messages:
            lines.append("\n=== Derniers échanges ===")
            for msg in last_5_messages[-N_LAST_MESSAGES:]:
                role    = msg.get("role", "user")
                content = msg.get("content", "").strip()
                prefix  = "Parent    :" if role == "user" else "Assistant :"
                # Tronquer si trop long
                if len(content) > 300:
                    content = content[:300] + "..."
                lines.append(f"{prefix} {content}")

        return "\n".join(lines)

    # ── MISE À JOUR APRÈS RÉPONSE ─────────────────────────────────────────────

    def update_after_response(self, last_5_messages: list,
                               summary: str,
                               keywords: list,
                               new_question: str,
                               new_answer: str,
                               total_messages: int) -> dict:
        """
        Met à jour le résumé et les mots-clés après chaque réponse LLM.
        À appeler APRÈS avoir obtenu la réponse, AVANT de sauvegarder en DB.

        Args:
            last_5_messages : 5 derniers messages (avant ce tour)
            summary         : résumé actuel depuis la DB
            keywords        : mots-clés actuels depuis la DB
            new_question    : question du parent (ce tour)
            new_answer      : réponse du LLM (ce tour)
            total_messages  : nombre total de messages dans la conversation

        Returns:
            {
                "summary"         : "nouveau résumé glissant...",
                "keywords"        : ["crise", "PECS", ...],
                "should_update_db": True | False
            }
        """
        # ── Règle : pas de résumé si <= 5 messages ────────────────────────
        if total_messages <= MIN_MESSAGES_FOR_SUMMARY:
            logger.info(f"total_messages={total_messages} ≤ {MIN_MESSAGES_FOR_SUMMARY} → pas de résumé")
            return {
                "summary"         : "",
                "keywords"        : [],
                "should_update_db": False,
            }

        # ── Générer nouveau résumé glissant ───────────────────────────────
        new_summary  = self._update_summary(summary, new_question, new_answer)

        # ── Mettre à jour les mots-clés ───────────────────────────────────
        new_keywords = self._update_keywords(
            keywords, new_question, new_answer
        )

        logger.info(
            f"Mémoire mise à jour ✔ | "
            f"résumé={len(new_summary)} chars | "
            f"keywords={len(new_keywords)}"
        )

        return {
            "summary"         : new_summary,
            "keywords"        : new_keywords,
            "should_update_db": True,
        }

    # ── RÉSUMÉ GLISSANT ───────────────────────────────────────────────────────

    def _update_summary(self, previous_summary: str,
                        new_question: str,
                        new_answer: str) -> str:
        """
        Met à jour le résumé glissant.
        Input LLM : résumé précédent + 2 derniers échanges (pas tout l'historique)
        """
        # ── Via LLM ───────────────────────────────────────────────────────
        if self.has_llm:
            try:
                prev = f"Résumé précédent :\n{previous_summary}\n\n" \
                       if previous_summary.strip() else ""

                prompt = (
                    "Tu es un assistant spécialisé TSA et déficience intellectuelle.\n"
                    "Mets à jour le résumé de la conversation en intégrant "
                    "le nouvel échange. Le résumé doit être concis (3-5 phrases), "
                    "en français, et capturer les points clés de la situation "
                    "de l'enfant et des préoccupations du parent.\n\n"
                    f"{prev}"
                    f"Nouvel échange :\n"
                    f"Parent    : {new_question[:500]}\n"
                    f"Assistant : {new_answer[:500]}\n\n"
                    "Nouveau résumé (3-5 phrases) :"
                )
                summary = self._llm_client.generate(prompt)
                return summary.strip()

            except Exception as e:
                logger.warning(f"Erreur LLM résumé : {e} → fallback")

        # ── Fallback : concaténation simple ───────────────────────────────
        new_entry = f"Parent : {new_question[:200]} | Réponse : {new_answer[:200]}"
        if previous_summary.strip():
            return f"{previous_summary.strip()} {new_entry}"
        return new_entry

    # ── MOTS-CLÉS ─────────────────────────────────────────────────────────────

    def _update_keywords(self, current_keywords: list,
                         new_question: str,
                         new_answer: str) -> list:
        """
        Met à jour les mots-clés cliniques.
        Combine : mots-clés existants + nouveaux extraits du dernier échange.
        """
        combined_text = f"{new_question} {new_answer}".lower()
        new_kws = set(current_keywords)

        # ── Via LLM ───────────────────────────────────────────────────────
        if self.has_llm:
            try:
                prompt = (
                    "Extrais les mots-clés cliniques importants du texte suivant "
                    "(troubles, comportements, besoins, méthodes, symptômes). "
                    "Retourne UNIQUEMENT une liste de mots séparés par des virgules, "
                    "sans explication, en français, maximum 10 mots-clés.\n\n"
                    f"Texte : {combined_text[:600]}\n\n"
                    "Mots-clés :"
                )
                result  = self._llm_client.generate(prompt)
                llm_kws = [k.strip().lower() for k in result.split(",")
                           if len(k.strip()) > 2]
                new_kws.update(llm_kws[:10])

            except Exception as e:
                logger.warning(f"Erreur LLM mots-clés : {e} → fallback règles")
                new_kws.update(self._extract_keywords_rules(combined_text))
        else:
            # Fallback : extraction par règles
            new_kws.update(self._extract_keywords_rules(combined_text))

        # Dédupliquer et limiter
        return list(new_kws)[:MAX_KEYWORDS]

    def _extract_keywords_rules(self, text: str) -> list:
        """Extrait les mots-clés cliniques par correspondance de liste."""
        text_lower = text.lower()
        found = []
        for kw in CLINICAL_KEYWORDS:
            if kw.lower() in text_lower:
                found.append(kw.lower())
        return found


# TEST

if __name__ == "__main__":

    mm = MemoryManager(llm_client=None)   # sans LLM → fallback règles

    # ── Simuler une conversation de 3 messages (≤5 → pas de résumé) ──────
    print("\n" + "="*60)
    print("  CAS 1 : 3 messages (≤ 5) → pas de résumé")
    print("="*60)

    result = mm.update_after_response(
        last_5_messages = [],
        summary         = "",
        keywords        = [],
        new_question    = "Mon enfant fait des crises tous les matins.",
        new_answer      = "Les crises matinales sont souvent liées aux transitions.",
        total_messages  = 3,
    )
    print(f"  should_update_db : {result['should_update_db']}")
    print(f"  summary          : '{result['summary']}'")
    print(f"  keywords         : {result['keywords']}")

    # ── Simuler une conversation de 7 messages (> 5 → résumé généré) ─────
    print("\n" + "="*60)
    print("  CAS 2 : 7 messages (> 5) → résumé glissant")
    print("="*60)

    last_5 = [
        {"role": "user",      "content": "Mon enfant ne parle pas encore."},
        {"role": "assistant", "content": "Le PECS est une méthode adaptée."},
        {"role": "user",      "content": "Il fait aussi des crises de colère."},
        {"role": "assistant", "content": "Pour les crises, maintenez un environnement calme."},
        {"role": "user",      "content": "Il a aussi des problèmes de sommeil."},
    ]

    result = mm.update_after_response(
        last_5_messages = last_5,
        summary         = "Enfant TSA de 5 ans, non verbal, crises fréquentes.",
        keywords        = ["TSA", "non-verbal", "crise"],
        new_question    = "Comment gérer l'hypersensibilité sensorielle ?",
        new_answer      = "Créez un espace sensoriel adapté avec lumières douces.",
        total_messages  = 7,
    )
    print(f"  should_update_db : {result['should_update_db']}")
    print(f"  Nouveau résumé   : {result['summary']}")
    print(f"  Mots-clés        : {result['keywords']}")

    # ── Bloc mémoire pour le prompt ───────────────────────────────────────
    print("\n" + "="*60)
    print("  BLOC MÉMOIRE → Prompt LLM")
    print("="*60)
    block = mm.build_memory_block(
        last_5_messages = last_5,
        summary         = result["summary"],
        keywords        = result["keywords"],
    )
    print(block)
    print("="*60)