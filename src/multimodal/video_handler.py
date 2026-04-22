import os, logging, tempfile, time, base64
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Nouveau SDK Google GenAI ──────────────────────────────────────────────────
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_OK = True
except ImportError:
    GEMINI_OK = False
    logger.warning("google-genai non installé → pip install google-genai")

# ── OpenCV ────────────────────────────────────────────────────────────────────
try:
    import cv2
    CV2_OK = True
except ImportError:
    CV2_OK = False
    logger.warning("opencv non installé → pip install opencv-python")

# ── Whisper (optionnel — nécessite ffmpeg système) ────────────────────────────
try:
    import whisper as whisper_lib
    WHISPER_OK = True
except ImportError:
    WHISPER_OK = False

# ── BLIP-2 (fallback local) ───────────────────────────────────────────────────
try:
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    import torch
    from PIL import Image
    BLIP2_OK = True
except ImportError:
    BLIP2_OK = False


# VIDEO HANDLER

class VideoHandler:

    GEMINI_MODEL = "gemini-2.0-flash"   # modèle actuel (non déprécié)

    def __init__(self):
        self.gemini_client = None
        self.blip2_model   = None
        self.blip2_proc    = None

        # Init Gemini
        if GEMINI_OK:
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                try:
                    self.gemini_client = genai.Client(api_key=api_key)
                    logger.info(f"Gemini {self.GEMINI_MODEL} prêt ✔")
                except Exception as e:
                    logger.warning(f"Gemini init échoué : {e}")

        logger.info("VideoHandler initialisé ✔")

    # POINT D'ENTRÉE PRINCIPAL

    def process(self, video_path: str, language: str = "fr") -> dict:
        """
        Analyse une vidéo et retourne une description textuelle.

        Args:
            video_path : chemin vers le fichier vidéo
            language   : langue de la description ("fr" | "ar" | "en")

        Returns:
            {
                "description"     : str,   # texte final
                "method_used"     : str,   # méthode utilisée
                "duration_sec"    : float,
                "frames_analyzed" : int,
                "success"         : bool,
            }
        """
        path = Path(video_path)
        if not path.exists():
            return self._error(f"Fichier introuvable : {video_path}")

        duration = self._get_duration(str(path))
        logger.info(f"Vidéo : {path.name} | durée : {duration:.1f}s")

        # ── [1] Gemini 2.0 Flash ─────────────────────────────────────────────
        if self.gemini_client:
            try:
                logger.info("Tentative Gemini 2.0 Flash...")
                result = self._gemini_describe(str(path), language)
                result["duration_sec"] = duration
                return result
            except Exception as e:
                logger.warning(f"Gemini échoué : {e} → essai Qwen2-VL")

        # ── [2] Qwen2-VL local ───────────────────────────────────────────────
        try:
            logger.info("Tentative Qwen2-VL local...")
            result = self._qwen_describe(str(path), language)
            result["duration_sec"] = duration
            return result
        except Exception as e:
            logger.warning(f"Qwen2-VL échoué : {e} → fallback BLIP-2")

        # ── [3] BLIP-2 fallback ──────────────────────────────────────────────
        logger.info("Fallback : BLIP-2 frames")
        result = self._blip2_describe(str(path), language)
        result["duration_sec"] = duration
        return result

    # [1] GEMINI 2.0 FLASH  — Upload + description vidéo entière

    def _gemini_describe(self, video_path: str, language: str) -> dict:
        """
        Envoie la vidéo directement à Gemini via l'API Files.
        Gemini comprend le mouvement, les actions, le contexte temporel.
        """
        lang_prompt = {
            "fr": "Décris cette vidéo en français",
            "ar": "صف هذا الفيديو باللغة العربية",
            "en": "Describe this video in English",
        }.get(language, "Describe this video in French")

        prompt = f"""
{lang_prompt}. Concentre-toi sur :
- Les personnes présentes et leurs actions
- Les objets et l'environnement
- La séquence temporelle des événements
- Les comportements observables (communication, motricité, interactions)
Sois précis et objectif. Réponds en {language}.
"""

        # Détecter le type MIME
        suffix  = Path(video_path).suffix.lower()
        mime_map = {
            ".mp4": "video/mp4",
            ".avi": "video/avi",
            ".mov": "video/quicktime",
            ".mkv": "video/x-matroska",
            ".webm": "video/webm",
            ".3gp": "video/3gpp",
        }
        mime_type = mime_map.get(suffix, "video/mp4")

        logger.info("Upload vidéo vers Gemini API Files...")

        # Upload du fichier
        with open(video_path, "rb") as f:
            video_file = self.gemini_client.files.upload(
                file=f,
                config=genai_types.UploadFileConfig(mime_type=mime_type),
            )

        # Attendre que le fichier soit prêt
        logger.info("Attente traitement Gemini...")
        max_wait = 60
        waited   = 0
        while waited < max_wait:
            file_info = self.gemini_client.files.get(name=video_file.name)
            if hasattr(file_info, "state"):
                state = str(file_info.state)
                if "ACTIVE" in state:
                    break
                if "FAILED" in state:
                    raise RuntimeError("Gemini : traitement vidéo échoué")
            time.sleep(3)
            waited += 3

        logger.info("Vidéo prête — génération description...")

        # Génération
        response = self.gemini_client.models.generate_content(
            model=self.GEMINI_MODEL,
            contents=[
                genai_types.Part.from_uri(
                    file_uri=video_file.uri,
                    mime_type=mime_type,
                ),
                prompt,
            ],
        )

        # Nettoyage du fichier uploadé
        try:
            self.gemini_client.files.delete(name=video_file.name)
        except Exception:
            pass

        description = response.text.strip()
        if not description:
            raise ValueError("Gemini a retourné une description vide")

        logger.info(f"Gemini ✔ — {len(description)} caractères")
        return {
            "description"     : description,
            "method_used"     : f"gemini-{self.GEMINI_MODEL}",
            "frames_analyzed" : -1,    # Gemini traite la vidéo entière
            "success"         : True,
        }

    # [2] QWEN2-VL LOCAL

    def _qwen_describe(self, video_path: str, language: str) -> dict:
        """Qwen2-VL sur frames clés extraites."""
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        model_name = "Qwen/Qwen2-VL-7B-Instruct"
        logger.info(f"Chargement {model_name}...")

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_name)

        lang_instr = {"fr": "en français", "ar": "بالعربية", "en": "in English"}.get(language, "en français")

        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "max_pixels": 360*420, "fps": 1.0},
                {"type": "text", "text": f"Décris cette vidéo {lang_instr}, actions et comportements observés."},
            ],
        }]

        text     = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_inp, vid_inp = process_vision_info(messages)
        inputs   = processor(text=[text], images=img_inp, videos=vid_inp,
                             padding=True, return_tensors="pt").to(model.device)
        out_ids  = model.generate(**inputs, max_new_tokens=512)
        trimmed  = [o[len(i):] for i, o in zip(inputs.input_ids, out_ids)]
        result   = processor.batch_decode(trimmed, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)[0]

        return {
            "description"     : result.strip(),
            "method_used"     : "qwen2-vl-7b",
            "frames_analyzed" : -1,
            "success"         : True,
        }

    # [3] BLIP-2 + WHISPER (fallback CPU)

    def _blip2_describe(self, video_path: str, language: str) -> dict:
        """
        Extrait des frames avec OpenCV, décrit chacune avec BLIP-2,
        puis fusionne en une description cohérente.
        Fonctionne sans ffmpeg système.
        """
        if not CV2_OK:
            return self._error("OpenCV requis : pip install opencv-python")

        # ── Extraction frames ────────────────────────────────────────────────
        frames, n_frames = self._extract_frames(video_path, max_frames=8)
        if not frames:
            return self._error("Impossible d'extraire les frames")

        # ── BLIP-2 description ───────────────────────────────────────────────
        descriptions = []

        if BLIP2_OK:
            descriptions = self._blip2_frames(frames)
        else:
            # Fallback sans BLIP-2 : description générique
            descriptions = [
                f"Frame {i+1}/{len(frames)} : scène vidéo"
                for i in range(len(frames))
            ]
            logger.warning("BLIP-2 non disponible — descriptions génériques")

        # ── Audio (optionnel — ne plante pas si ffmpeg absent) ───────────────
        audio_text = self._extract_audio_safe(video_path)

        # ── Fusion descriptions ──────────────────────────────────────────────
        combined = self._fuse_descriptions(descriptions, audio_text, language)

        logger.info(f"BLIP-2 ✔ — {len(frames)} frames analysées")
        return {
            "description"     : combined,
            "method_used"     : "blip2+frames",
            "frames_analyzed" : len(frames),
            "success"         : bool(combined and combined != "Impossible d'analyser cette vidéo."),
        }

    def _blip2_frames(self, frames: list) -> list:
        """Décrit chaque frame avec BLIP-2."""
        import torch
        from PIL import Image

        if not self.blip2_model:
            logger.info("Chargement BLIP-2...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.blip2_proc  = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)

        device       = next(self.blip2_model.parameters()).device
        descriptions = []

        for i, frame_bgr in enumerate(frames):
            try:
                # BGR → RGB → PIL
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_img   = Image.fromarray(frame_rgb)

                inputs = self.blip2_proc(
                    images=pil_img,
                    text="Describe what you see in detail:",
                    return_tensors="pt",
                ).to(device)

                out = self.blip2_model.generate(**inputs, max_new_tokens=80)
                desc = self.blip2_proc.decode(out[0], skip_special_tokens=True).strip()
                descriptions.append(f"[{i+1}] {desc}")
                logger.info(f"  Frame {i+1}/{len(frames)} : {desc[:60]}...")
            except Exception as e:
                logger.warning(f"  Frame {i+1} échouée : {e}")
                descriptions.append(f"[{i+1}] frame non analysée")

        return descriptions

    def _fuse_descriptions(self, descriptions: list,
                            audio_text: str, language: str) -> str:
        """
        Fusionne les descriptions de frames en un texte cohérent.
        Sans LLM → simple concaténation structurée.
        """
        if not descriptions:
            return "Impossible d'analyser cette vidéo."

        # Dédupliquer les descriptions similaires
        unique = []
        for d in descriptions:
            content = d.split("] ", 1)[-1].strip()
            if not unique or content.lower() != unique[-1].lower():
                unique.append(content)

        parts = []

        if language == "fr":
            parts.append("Description de la vidéo :")
            for i, u in enumerate(unique):
                parts.append(f"  • Séquence {i+1} : {u}")
            if audio_text:
                parts.append(f"\nAudio détecté : {audio_text}")
        elif language == "ar":
            parts.append("وصف الفيديو:")
            for i, u in enumerate(unique):
                parts.append(f"  • مشهد {i+1}: {u}")
            if audio_text:
                parts.append(f"\nالصوت المكتشف: {audio_text}")
        else:
            parts.append("Video description:")
            for i, u in enumerate(unique):
                parts.append(f"  • Scene {i+1}: {u}")
            if audio_text:
                parts.append(f"\nDetected audio: {audio_text}")

        return "\n".join(parts)

    # UTILITAIRES

    def _extract_frames(self, video_path: str,
                         max_frames: int = 8) -> tuple:
        """Extrait max_frames frames réparties uniformément."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Impossible d'ouvrir : {video_path}")
            return [], 0

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25

        if total <= 0:
            cap.release()
            return [], 0

        # Positions des frames à extraire (uniformément réparties)
        step   = max(1, total // max_frames)
        positions = [i * step for i in range(max_frames) if i * step < total]

        frames = []
        for pos in positions:
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
            ret, frame = cap.read()
            if ret and frame is not None:
                # Redimensionner si trop grande
                h, w = frame.shape[:2]
                if w > 640:
                    scale = 640 / w
                    frame = cv2.resize(frame, (640, int(h * scale)))
                frames.append(frame)

        cap.release()
        logger.info(f"  {len(frames)} frames extraites sur {total} total")
        return frames, total

    def _get_duration(self, video_path: str) -> float:
        """Durée de la vidéo via OpenCV (sans ffmpeg)."""
        if not CV2_OK:
            return 0.0
        try:
            cap   = cv2.VideoCapture(video_path)
            fps   = cap.get(cv2.CAP_PROP_FPS) or 25
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return round(total / fps, 2) if fps > 0 else 0.0
        except Exception:
            return 0.0

    def _extract_audio_safe(self, video_path: str) -> str:
        """
        Tente d'extraire l'audio SANS ffmpeg système.
        Retourne "" si ffmpeg absent (ne plante pas).
        """
        if not WHISPER_OK:
            return ""

        try:
            import subprocess
            # Vérifier si ffmpeg est disponible
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True, timeout=5
            )
            if result.returncode != 0:
                logger.info("ffmpeg non disponible — audio ignoré")
                return ""
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.info("ffmpeg non trouvé — audio ignoré (vidéo analysée sans son)")
            return ""

        # ffmpeg disponible → extraire audio
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            import subprocess
            subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1", tmp_path,
            ], capture_output=True, timeout=30)

            model  = whisper_lib.load_model("tiny")
            result = model.transcribe(tmp_path, language=None)
            text   = result.get("text", "").strip()
            Path(tmp_path).unlink(missing_ok=True)
            return text

        except Exception as e:
            logger.warning(f"Extraction audio échouée : {e}")
            return ""

    def _error(self, message: str) -> dict:
        logger.error(message)
        return {
            "description"     : message,
            "method_used"     : "error",
            "duration_sec"    : 0.0,
            "frames_analyzed" : 0,
            "success"         : False,
        }


# TEST EN LIGNE DE COMMANDE

if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    if len(sys.argv) < 2:
        print("Usage : python video_handler.py <chemin_video.mp4>")
        sys.exit(1)

    handler = VideoHandler()
    result  = handler.process(sys.argv[1], language="fr")

    print("\n" + "="*60)
    print(f"  Méthode  : {result['method_used']}")
    print(f"  Durée    : {result['duration_sec']}s")
    print(f"  Frames   : {result['frames_analyzed']}")
    print(f"  Succès   : {result['success']}")
    print("="*60)
    print(result["description"])
    print("="*60)