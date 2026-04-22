import os, logging, shutil, sys
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ── Imports optionnels ────────────────────────────────────────────────────────
try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False
    logger.warning("pillow non installé → pip install pillow")

try:
    import pytesseract
    # Windows : chemin par défaut de Tesseract
    if sys.platform == "win32":
        _tess_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe"),
        ]
        for p in _tess_paths:
            if os.path.exists(p):
                pytesseract.pytesseract.tesseract_cmd = p
                break
    TESSERACT_OK = True
except ImportError:
    TESSERACT_OK = False
    logger.warning("pytesseract non installé → pip install pytesseract")

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch
    BLIP_OK = True
except ImportError:
    BLIP_OK = False
    logger.info("transformers non installé → description BLIP désactivée")

try:
    from language_handler import LanguageHandler
    LANG_OK = True
except ImportError:
    LANG_OK = False
    logger.warning("language_handler.py introuvable")


# CONFIGURATION

# Seuil mots OCR : si >= ce seuil → mode OCR, sinon → mode Description
OCR_MIN_WORDS = 10

# Langues OCR Tesseract (fra=français, ara=arabe, eng=anglais)
OCR_LANG = "fra+ara+eng"

# Formats images supportés
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# Modèle BLIP pour description locale (téléchargé automatiquement ~1GB)
BLIP_MODEL = "Salesforce/blip-image-captioning-large"
HF_TOKEN = os.getenv("HF_TOKEN")


# CLASSE PRINCIPALE

class ImageHandler:
    """
    Extrait du texte depuis une image (OCR ou description visuelle).

    Modes :
        "ocr"         → document, ordonnance, screenshot, rapport
        "description" → photo d'enfant, dessin, image sans texte
        "auto"        → détection automatique (défaut)
    """

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client : client LLM avec vision (ex: GPT-4o, Claude)
                         Si None → BLIP local utilisé pour la description
        """
        self._llm_client = llm_client
        self.lang_handler = LanguageHandler() if LANG_OK else None

        # Charger BLIP si disponible et pas de LLM vision
        self._blip_processor = None
        self._blip_model      = None
        if BLIP_OK and self._llm_client is None:
            self._load_blip()

        logger.info("ImageHandler initialisé ✔")

    # ── LLM VISION ────────────────────────────────────────────────────────────

    def set_llm(self, llm_client) -> None:
        """
        Injecte un client LLM avec capacité vision (ex: GPT-4o).
        Prioritaire sur BLIP pour la description d'images.

        Exemple :
            handler.set_llm(mon_llm)
        """
        self._llm_client = llm_client
        logger.info("LLM vision défini ✔")

    @property
    def has_llm(self) -> bool:
        return self._llm_client is not None

    # ── BLIP (description locale) ─────────────────────────────────────────────

    def _load_blip(self):
        """Charge le modèle BLIP en mémoire (téléchargé si absent)."""
        try:
            logger.info(f"Chargement BLIP ({BLIP_MODEL})...")
            self._blip_processor = BlipProcessor.from_pretrained(BLIP_MODEL)
            self._blip_model = BlipForConditionalGeneration.from_pretrained(
                BLIP_MODEL,
                torch_dtype=torch.float32
            )
            logger.info("BLIP chargé ✔")
        except Exception as e:
            logger.warning(f"BLIP non disponible : {e}")
            self._blip_processor = None
            self._blip_model      = None

    # ── MODE OCR ──────────────────────────────────────────────────────────────

    def extract_ocr(self, image: "Image.Image") -> str:
        """
        Extrait le texte d'une image via Tesseract OCR.

        Args:
            image : objet PIL Image

        Returns:
            Texte extrait (str)
        """
        if not TESSERACT_OK:
            raise ImportError(
                "pytesseract non installé → pip install pytesseract\n"
                "Windows : https://github.com/UB-Mannheim/tesseract/wiki"
            )

        # Prétraitement : convertir en niveaux de gris pour meilleur OCR
        img_gray = image.convert("L")

        try:
            text = pytesseract.image_to_string(img_gray, lang=OCR_LANG)
            return text.strip()
        except pytesseract.TesseractNotFoundError:
            raise EnvironmentError(
                "Tesseract introuvable.\n"
                "  Windows : https://github.com/UB-Mannheim/tesseract/wiki\n"
                "  Linux   : sudo apt install tesseract-ocr tesseract-ocr-fra "
                "tesseract-ocr-ara tesseract-ocr-eng"
            )

    # ── MODE DESCRIPTION ──────────────────────────────────────────────────────

    def describe_image(self, image: "Image.Image", image_path: str = "") -> str:
        """
        Génère une description textuelle d'une image.

        Priorité :
            1. LLM vision (si set_llm() appelé)
            2. BLIP local (si transformers installé)
            3. Description basique (fallback)

        Args:
            image      : objet PIL Image
            image_path : chemin vers le fichier (pour LLM vision)

        Returns:
            Description en français
        """

        # ── 1. LLM Vision (GPT-4o, Claude Haiku, etc.) ───────────────────
        if self.has_llm:
            try:
                import base64
                with open(image_path, "rb") as f:
                    img_b64 = base64.b64encode(f.read()).decode("utf-8")
                ext = os.path.splitext(image_path)[1].lower().replace(".", "")
                media_type = f"image/{ext}" if ext != "jpg" else "image/jpeg"

                prompt = (
                    "Tu es un assistant spécialisé TSA et déficience intellectuelle. "
                    "Décris cette image de façon précise et concise en français, "
                    "en te concentrant sur les éléments pertinents pour comprendre "
                    "la situation d'un enfant ou d'un parent."
                )
                description = self._llm_client.generate_vision(
                    prompt=prompt,
                    image_b64=img_b64,
                    media_type=media_type,
                )
                logger.info("Description via LLM vision ✔")
                return description.strip()
            except Exception as e:
                logger.warning(f"Erreur LLM vision : {e} → fallback BLIP")

        # ── 2. BLIP local ─────────────────────────────────────────────────
        if self._blip_model is not None and self._blip_processor is not None:
            try:
                inputs = self._blip_processor(image, return_tensors="pt")
                with torch.no_grad():
                    out = self._blip_model.generate(**inputs, max_new_tokens=100)
                caption_en = self._blip_processor.decode(
                    out[0], skip_special_tokens=True
                )
                # Traduire la description anglaise en français
                if self.lang_handler:
                    caption_fr = self.lang_handler.translate_to_internal(
                        caption_en, "en"
                    )
                else:
                    caption_fr = caption_en
                logger.info("Description via BLIP ✔")
                return caption_fr
            except Exception as e:
                logger.warning(f"Erreur BLIP : {e} → fallback basique")

        # ── 3. Fallback basique ───────────────────────────────────────────
        w, h = image.size
        mode = image.mode
        logger.info("Description basique (fallback)")
        return (
            f"Image reçue ({w}x{h} pixels, mode {mode}). "
            "Aucun texte détecté et description visuelle non disponible. "
            "Veuillez décrire le contenu de l'image en texte."
        )

    # ── DÉTECTION AUTOMATIQUE DU MODE ─────────────────────────────────────────

    def detect_mode(self, image: "Image.Image") -> str:
        """
        Détermine automatiquement si l'image contient du texte (OCR)
        ou si c'est une image visuelle (description).

        Returns:
            "ocr" | "description"
        """
        if not TESSERACT_OK:
            return "description"

        try:
            sample_text = pytesseract.image_to_string(
                image.convert("L"), lang=OCR_LANG
            )
            word_count = len(sample_text.split())
            mode = "ocr" if word_count >= OCR_MIN_WORDS else "description"
            logger.info(f"Mode détecté : {mode} ({word_count} mots OCR)")
            return mode
        except Exception:
            return "description"

    # ── PIPELINE PRINCIPAL ────────────────────────────────────────────────────

    def process(self, image_path: str, mode: str = "auto") -> dict:
        """
        Pipeline complet : image → texte extrait + traduction en français.

        Args:
            image_path : chemin vers le fichier image
            mode       : "auto" | "ocr" | "description"

        Returns:
            {
                "status"           : "success" | "error",
                "image_path"       : "photo.jpg",
                "mode"             : "ocr" | "description",
                "extracted_text"   : "texte brut extrait...",
                "detected_lang"    : "fr",
                "lang_name"        : "Français",
                "needs_translation": False,
                "translated_text"  : "texte en français...",  ← envoyer au RAG
                "word_count"       : 38,
                "error"            : None,
            }
        """
        result = {
            "status"           : "error",
            "image_path"       : image_path,
            "mode"             : mode,
            "extracted_text"   : "",
            "detected_lang"    : "fr",
            "lang_name"        : "Français",
            "needs_translation": False,
            "translated_text"  : "",
            "word_count"       : 0,
            "error"            : None,
        }

        # Vérifier le fichier
        if not os.path.exists(image_path):
            result["error"] = f"Fichier introuvable : {image_path}"
            logger.error(result["error"])
            return result

        ext = os.path.splitext(image_path)[1].lower()
        if ext not in SUPPORTED_FORMATS:
            result["error"] = f"Format non supporté : {ext}"
            logger.error(result["error"])
            return result

        if not PIL_OK:
            result["error"] = "pillow non installé → pip install pillow"
            return result

        try:
            # Charger l'image
            image = Image.open(image_path).convert("RGB")

            # Détecter le mode si auto
            if mode == "auto":
                mode = self.detect_mode(image)
            result["mode"] = mode

            # Extraction selon le mode
            if mode == "ocr":
                raw_text = self.extract_ocr(image)
            else:
                raw_text = self.describe_image(image, image_path)

            if not raw_text or len(raw_text.strip()) < 5:
                result["error"] = "Aucun contenu extrait de l'image"
                return result

            result["extracted_text"] = raw_text

            # Détection langue + Traduction
            if self.lang_handler:
                lang_result = self.lang_handler.process(raw_text)
                result["detected_lang"]    = lang_result["detected_lang"]
                result["lang_name"]        = lang_result["lang_name"]
                result["needs_translation"]= lang_result["needs_translation"]
                result["translated_text"]  = lang_result["translated_text"]
            else:
                result["translated_text"] = raw_text

            result["word_count"] = len(result["translated_text"].split())
            result["status"]     = "success"

            logger.info(
                f"process() ✔ | mode={mode} | "
                f"langue={result['detected_lang']} | "
                f"mots={result['word_count']}"
            )

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Erreur ImageHandler : {e}")

        return result


# TEST

if __name__ == "__main__":
    handler    = ImageHandler()
    image_file = sys.argv[1] if len(sys.argv) > 1 else "data/test/test_image.png"

    if not os.path.exists(image_file):
        print(f"\n  Fichier '{image_file}' introuvable.")
        print("Usage : python image_handler.py chemin/vers/image.png\n")
        sys.exit(1)

    print("\n" + "="*60)
    print("  TEST — Image → Texte")
    print("="*60)
    print(f"  Fichier : {image_file}\n")

    result = handler.process(image_file)

    if result["status"] == "success":
        print(f"  ✔ Mode            : {result['mode']}")
        print(f"  Langue            : {result['lang_name']} ({result['detected_lang']})")
        print(f"  Traduction        : {result['needs_translation']}")
        print(f"  Mots              : {result['word_count']}")
        print(f"\n  Texte extrait :")
        print(f"  {result['extracted_text'][:300]}")
        if result["needs_translation"]:
            print(f"\n  Texte FR (RAG) :")
            print(f"  {result['translated_text'][:300]}")
    else:
        print(f"  ✘ Erreur : {result['error']}")

    print("\n" + "="*60)