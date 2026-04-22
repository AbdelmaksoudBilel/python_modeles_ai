from lingua import Language, LanguageDetectorBuilder
from deep_translator import GoogleTranslator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# CONFIGURATION

# Langue interne du système (vers laquelle tout est traduit)
INTERNAL_LANG = "fr"

# Langues supportées par l'application
SUPPORTED_LANGUAGES = {
    Language.FRENCH : "fr",
    Language.ARABIC  : "ar",
    Language.ENGLISH : "en",
}

# Noms lisibles pour les logs et messages
LANG_NAMES = {
    "fr": "Français",
    "ar": "Arabe",
    "en": "Anglais",
}

# Codes Google Translate (deep-translator)
GOOGLE_CODES = {
    "fr": "fr",
    "ar": "ar",
    "en": "en",
}


# CLASSE PRINCIPALE

class LanguageHandler:
    """
    Gère la détection de langue et la traduction vers le français.

    Workflow :
        Input parent (fr / ar / en)
            ↓
        detect_language()  → code ISO ("fr", "ar", "en")
            ↓
        translate_to_internal()  → texte en français
            ↓
        process()  → dict complet avec tout
    """

    def __init__(self):
        # Construire le détecteur avec seulement les 3 langues ciblées
        # (plus précis et plus rapide qu'avec toutes les langues)
        logger.info("Chargement du détecteur de langue (lingua)...")
        self.detector = LanguageDetectorBuilder.from_languages(
            Language.FRENCH,
            Language.ARABIC,
            Language.ENGLISH,
        ).with_preloaded_language_models().build()

        logger.info("LanguageHandler initialisé ✔")

    # ── DÉTECTION ─────────────────────────────────────────────────────────────

    def detect_language(self, text: str) -> str:
        """
        Détecte la langue d'un texte.

        Args:
            text : texte du parent

        Returns:
            Code ISO : "fr" | "ar" | "en"
            Retourne "fr" par défaut si la détection échoue.
        """
        if not text or len(text.strip()) < 3:
            logger.warning("Texte trop court pour la détection → défaut : fr")
            return "fr"

        try:
            detected = self.detector.detect_language_of(text)

            if detected is None:
                logger.warning("Langue non détectée → défaut : fr")
                return "fr"

            lang_code = SUPPORTED_LANGUAGES.get(detected, "fr")
            confidence = self._get_confidence(text, detected)

            logger.info(
                f"Langue détectée : {LANG_NAMES.get(lang_code, lang_code)} "
                f"(confiance : {confidence:.2f})"
            )
            return lang_code

        except Exception as e:
            logger.error(f"Erreur détection langue : {e} → défaut : fr")
            return "fr"

    def _get_confidence(self, text: str, language: Language) -> float:
        """Retourne le score de confiance pour une langue donnée."""
        try:
            scores = self.detector.compute_language_confidence_values(text)
            for result in scores:
                if result.language == language:
                    return result.value
        except Exception:
            pass
        return 0.0

    def detect_with_confidence(self, text: str) -> dict:
        """
        Détecte la langue avec le score de confiance.

        Returns:
            {
                "lang"      : "fr",
                "confidence": 0.97,
                "all_scores": {"fr": 0.97, "en": 0.02, "ar": 0.01}
            }
        """
        if not text or len(text.strip()) < 3:
            return {"lang": "fr", "confidence": 1.0, "all_scores": {}}

        try:
            scores = self.detector.compute_language_confidence_values(text)
            all_scores = {}
            best_lang  = "fr"
            best_score = 0.0

            for result in scores:
                code = SUPPORTED_LANGUAGES.get(result.language)
                if code:
                    all_scores[code] = round(result.value, 4)
                    if result.value > best_score:
                        best_score = result.value
                        best_lang  = code

            return {
                "lang"      : best_lang,
                "confidence": round(best_score, 4),
                "all_scores": all_scores,
            }

        except Exception as e:
            logger.error(f"Erreur détection avec confiance : {e}")
            return {"lang": "fr", "confidence": 0.0, "all_scores": {}}

    # ── TRADUCTION ────────────────────────────────────────────────────────────

    def translate_to_internal(self, text: str, source_lang: str) -> str:
        """
        Traduit le texte vers la langue interne (français).

        Args:
            text        : texte à traduire
            source_lang : code ISO de la langue source ("ar", "en", "fr")

        Returns:
            Texte traduit en français.
            Retourne le texte original si la traduction échoue.
        """
        if source_lang == INTERNAL_LANG:
            return text  # déjà en français

        if not text or len(text.strip()) == 0:
            return text

        try:
            src = GOOGLE_CODES.get(source_lang, source_lang)
            tgt = GOOGLE_CODES[INTERNAL_LANG]

            translated = GoogleTranslator(source=src, target=tgt).translate(text)

            if not translated:
                logger.warning("Traduction vide → texte original conservé")
                return text

            logger.info(
                f"Traduction {LANG_NAMES.get(source_lang, source_lang)} → "
                f"{LANG_NAMES[INTERNAL_LANG]} ✔"
            )
            return translated

        except Exception as e:
            logger.error(f"Erreur traduction : {e} → texte original conservé")
            return text

    def translate_response_to_parent(self, text: str, target_lang: str) -> str:
        """
        Traduit la réponse du LLM (en français) vers la langue du parent.

        Args:
            text        : réponse en français générée par le LLM
            target_lang : langue du parent ("ar", "en", "fr")

        Returns:
            Réponse traduite dans la langue du parent.
        """
        if target_lang == INTERNAL_LANG:
            return text  # déjà en français

        if not text or len(text.strip()) == 0:
            return text

        try:
            src = GOOGLE_CODES[INTERNAL_LANG]
            tgt = GOOGLE_CODES.get(target_lang, target_lang)

            translated = GoogleTranslator(source=src, target=tgt).translate(text)

            if not translated:
                logger.warning("Traduction réponse vide → texte original conservé")
                return text

            logger.info(
                f"Réponse traduite : {LANG_NAMES[INTERNAL_LANG]} → "
                f"{LANG_NAMES.get(target_lang, target_lang)} ✔"
            )
            return translated

        except Exception as e:
            logger.error(f"Erreur traduction réponse : {e} → texte original conservé")
            return text

    # ── PIPELINE COMPLET ──────────────────────────────────────────────────────

    def process(self, text: str) -> dict:
        """
        Pipeline complet : détection + traduction vers français.

        Args:
            text : message brut du parent

        Returns:
            {
                "original_text"    : "My child doesn't speak",
                "detected_lang"    : "en",
                "lang_name"        : "Anglais",
                "confidence"       : 0.97,
                "needs_translation": True,
                "translated_text"  : "Mon enfant ne parle pas",
                "internal_lang"    : "fr",
            }
        """
        if not text:
            return {
                "original_text"    : "",
                "detected_lang"    : "fr",
                "lang_name"        : "Français",
                "confidence"       : 1.0,
                "needs_translation": False,
                "translated_text"  : "",
                "internal_lang"    : INTERNAL_LANG,
            }

        # ── Étape 1 : Détection ───────────────────────────────────────────
        detection         = self.detect_with_confidence(text)
        detected_lang     = detection["lang"]
        confidence        = detection["confidence"]
        needs_translation = detected_lang != INTERNAL_LANG

        # ── Étape 2 : Traduction ──────────────────────────────────────────
        if needs_translation:
            translated_text = self.translate_to_internal(text, detected_lang)
        else:
            translated_text = text

        return {
            "original_text"    : text,
            "detected_lang"    : detected_lang,
            "lang_name"        : LANG_NAMES.get(detected_lang, detected_lang),
            "confidence"       : confidence,
            "needs_translation": needs_translation,
            "translated_text"  : translated_text,
            "internal_lang"    : INTERNAL_LANG,
        }


# TEST RAPIDE

if __name__ == "__main__":

    handler = LanguageHandler()

    test_inputs = [
        # Français
        "Mon enfant ne parle pas encore, il a 3 ans. Que puis-je faire ?",
        # Anglais
        "My child has autism and refuses to eat. What should I do?",
        # Arabe
        "طفلي لا يتحدث ولا يستجيب عند مناداته. هل هذا مؤشر على التوحد؟",
        # Texte court
        "Bonjour",
    ]

    print("\n" + "="*60)
    print("  TEST — Détection de langue + Traduction")
    print("="*60)

    for i, text in enumerate(test_inputs, 1):
        print(f"\n[Test {i}]")
        print(f"  Input    : {text[:60]}{'...' if len(text) > 60 else ''}")

        result = handler.process(text)

        print(f"  Langue   : {result['lang_name']} ({result['detected_lang']}) "
              f"— confiance : {result['confidence']:.2f}")
        print(f"  Traduit  : {result['needs_translation']}")
        if result["needs_translation"]:
            print(f"  Résultat : {result['translated_text'][:80]}...")

    # ── Test traduction réponse vers parent ───────────────────────────────
    print("\n" + "-"*60)
    print("  TEST — Traduction réponse LLM → langue parent")
    print("-"*60)

    llm_response = (
        "Il est important de mettre en place une routine structurée "
        "pour votre enfant. Consultez un spécialiste TSA pour un suivi adapté."
    )

    for lang in ["en", "ar", "fr"]:
        translated = handler.translate_response_to_parent(llm_response, lang)
        print(f"\n  [{LANG_NAMES[lang]}]")
        print(f"  {translated[:100]}")

    print("\n" + "="*60)
    print("  ✔ Tests terminés")
    print("="*60)