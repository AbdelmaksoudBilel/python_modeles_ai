import os, logging, shutil, sys
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Injection ffmpeg Windows (winget installe dans WinGet\Links) ──────────────
if sys.platform == "win32":
    _links = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Links")
    if os.path.isdir(_links) and shutil.which("ffmpeg") is None:
        os.environ["PATH"] = _links + os.pathsep + os.environ["PATH"]
        logger.info(f"ffmpeg PATH injecté : {_links}")

# ── Imports optionnels ────────────────────────────────────────────────────────
try:
    import whisper
    WHISPER_OK = True
except ImportError:
    WHISPER_OK = False
    logger.warning("whisper non installé → pip install openai-whisper")

try:
    from language_handler import LanguageHandler
    LANG_OK = True
except ImportError:
    LANG_OK = False
    logger.warning("language_handler.py introuvable")


# CONFIGURATION

WHISPER_MODEL_SIZE = "base"       # tiny | base | small | medium
MAX_AUDIO_DURATION = 120          # secondes max (0 = illimité)
SUPPORTED_FORMATS  = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".webm", ".opus"}


# CLASSE PRINCIPALE

class SpeechHandler:
    """
    Transforme un message vocal en texte traduit en français.

    Différence avec VideoHandler :
        - Pas d'extraction audio (le fichier EST déjà de l'audio)
        - Plus léger, plus rapide
        - Formats : mp3, wav, ogg, m4a, flac (WhatsApp envoie en .ogg/.opus)
    """

    def __init__(self, whisper_model_size: str = WHISPER_MODEL_SIZE):
        if WHISPER_OK:
            logger.info(f"Chargement Whisper ({whisper_model_size})...")
            self.whisper_model = whisper.load_model(whisper_model_size)
            logger.info("Whisper ✔")
        else:
            self.whisper_model = None

        self.lang_handler = LanguageHandler() if LANG_OK else None
        logger.info("SpeechHandler initialisé ✔")

    # ── VÉRIFICATION FFMPEG ───────────────────────────────────────────────────

    def _check_ffmpeg(self):
        """Vérifie que ffmpeg est accessible, lève une erreur claire sinon."""
        if shutil.which("ffmpeg") is None:
            raise EnvironmentError(
                "ffmpeg introuvable. Exécutez :\n"
                "  Windows : winget install ffmpeg  puis redémarrez le terminal\n"
                "  Linux   : sudo apt install ffmpeg"
            )

    # ── TRANSCRIPTION ─────────────────────────────────────────────────────────

    def transcribe(self, audio_path: str) -> dict:
        """
        Transcrit un fichier audio → texte brut via Whisper.

        Args:
            audio_path : chemin vers le fichier audio

        Returns:
            { "text": "...", "language": "fr", "duration": 12.4 }
        """
        if not WHISPER_OK or self.whisper_model is None:
            raise ImportError("pip install openai-whisper")

        self._check_ffmpeg()

        # Vérifier le format
        ext = os.path.splitext(audio_path)[1].lower()
        if ext not in SUPPORTED_FORMATS:
            raise ValueError(
                f"Format non supporté : {ext}  |  "
                f"Acceptés : {', '.join(SUPPORTED_FORMATS)}"
            )

        # Vérifier la durée si possible
        if MAX_AUDIO_DURATION > 0:
            try:
                import wave
                if ext == ".wav":
                    with wave.open(audio_path, 'r') as f:
                        duration = f.getnframes() / f.getframerate()
                    if duration > MAX_AUDIO_DURATION:
                        logger.warning(
                            f"Audio long ({duration:.0f}s) — "
                            f"traitement peut prendre du temps"
                        )
            except Exception:
                pass   # durée non critique

        logger.info(f"Transcription : {os.path.basename(audio_path)}...")
        result = self.whisper_model.transcribe(
            audio_path,
            task="transcribe",
            fp16=False,
            verbose=False,
        )

        text = result.get("text", "").strip()
        lang = result.get("language", "fr")

        logger.info(f"Transcription ✔ — langue={lang} — {len(text)} chars")
        return {
            "text"    : text,
            "language": lang,
            "segments": result.get("segments", []),
        }

    # ── PIPELINE PRINCIPAL ────────────────────────────────────────────────────

    def process(self, audio_path: str) -> dict:
        """
        Pipeline complet : audio → transcription + traduction en français.

        Args:
            audio_path : chemin vers le fichier audio

        Returns:
            {
                "status"           : "success" | "error",
                "audio_path"       : "voice.ogg",
                "transcription"    : "texte original...",
                "detected_lang"    : "ar",
                "lang_name"        : "Arabe",
                "needs_translation": True,
                "translated_text"  : "texte en français...",  ← envoyer au RAG
                "word_count"       : 45,
                "error"            : None,
            }
        """
        result = {
            "status"           : "error",
            "audio_path"       : audio_path,
            "transcription"    : "",
            "detected_lang"    : "fr",
            "lang_name"        : "Français",
            "needs_translation": False,
            "translated_text"  : "",
            "word_count"       : 0,
            "error"            : None,
        }

        # Vérifier le fichier
        if not os.path.exists(audio_path):
            result["error"] = f"Fichier introuvable : {audio_path}"
            logger.error(result["error"])
            return result

        try:
            # 1. Transcription Whisper
            transcription = self.transcribe(audio_path)
            raw_text      = transcription["text"]

            if not raw_text:
                result["error"] = "Transcription vide — aucune parole détectée"
                return result

            result["transcription"] = raw_text

            # 2. Détection langue + Traduction
            if self.lang_handler:
                lang_result = self.lang_handler.process(raw_text)
                result["detected_lang"]    = lang_result["detected_lang"]
                result["lang_name"]        = lang_result["lang_name"]
                result["needs_translation"]= lang_result["needs_translation"]
                result["translated_text"]  = lang_result["translated_text"]
            else:
                # Fallback : utiliser la langue détectée par Whisper
                result["detected_lang"]   = transcription["language"]
                result["translated_text"] = raw_text

            result["word_count"] = len(result["translated_text"].split())
            result["status"]     = "success"

            logger.info(
                f"process() ✔ | langue={result['detected_lang']} | "
                f"mots={result['word_count']}"
            )

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Erreur SpeechHandler : {e}")

        return result


# TEST

if __name__ == "__main__":
    import sys

    handler    = SpeechHandler()
    audio_file = sys.argv[1] if len(sys.argv) > 1 else "data/test/test_audio.mp3"

    if not os.path.exists(audio_file):
        print(f"\n⚠️  Fichier '{audio_file}' introuvable.")
        print("Usage : python speech_handler.py chemin/vers/audio.ogg\n")
        sys.exit(1)

    print("\n" + "="*60)
    print("  TEST — Message Vocal → Texte")
    print("="*60)
    print(f"  Fichier : {audio_file}\n")

    result = handler.process(audio_file)

    if result["status"] == "success":
        print(f"  ✔ Langue          : {result['lang_name']} ({result['detected_lang']})")
        print(f"  Traduction        : {result['needs_translation']}")
        print(f"  Mots              : {result['word_count']}")
        print(f"\n  Transcription :")
        print(f"  {result['transcription'][:300]}")
        if result["needs_translation"]:
            print(f"\n  Texte FR (RAG) :")
            print(f"  {result['translated_text'][:300]}")
    else:
        print(f"  ✘ Erreur : {result['error']}")

    print("\n" + "="*60)