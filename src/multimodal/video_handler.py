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

# Note: Whisper removed — audio transcription is disabled (video-only analysis)

# ── LLM client (Groq) pour fallback léger vision → texte
try:
    from ..llm.llm_client import LLMClient
    LLMCLIENT_OK = True
except Exception:
    LLMCLIENT_OK = False


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
                logger.warning(f"Gemini échoué : {e} → fallback Groq vision")

        # ── [2] Light vision fallback (Groq) — avoids large local VLMs
        try:
            logger.info("Tentative fallback Groq Vision...")
            result = self._groq_vision_fallback(str(path), language)
            result["duration_sec"] = duration
            return result
        except Exception as e:
            logger.warning(f"Fallback Groq échoué : {e} — retour erreur")
            return self._error(f"Analyse vidéo impossible : {e}")

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

    def _groq_vision_fallback(self, video_path: str, language: str) -> dict:
        """Lightweight fallback: extract up to 2 frames, send as base64 to Groq Vision LLM.

        Uses `LLMClient` (Groq) if available; otherwise returns a simple
        concatenation of frame placeholders.
        """
        frames, n_frames = self._extract_frames(video_path, max_frames=2)
        if not frames:
            return self._error("Impossible d'extraire les frames")

        # Encode frames as JPEG base64
        imgs_b64 = []
        for frame in frames:
            try:
                # Ensure frames are small (see _extract_frames resizing)
                # Compress aggressively to reduce token usage
                _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                b64 = base64.b64encode(buf.tobytes()).decode('ascii')
                imgs_b64.append(b64)
            except Exception:
                imgs_b64.append("")

        # Build a compact prompt for Groq
        prompt_parts = [
            f"You are a helpful assistant. Describe the scene, actions, objects and any observable behaviors in these images. Respond in {language}.",
            "Provide a concise, objective summary (2-4 sentences) and list notable actions/objects.",
        ]
        for i, b in enumerate(imgs_b64):
            prompt_parts.append(f"---IMAGE {i+1} (base64)---\n{b}")

        prompt = "\n\n".join(prompt_parts)

        # Call Groq via LLMClient if available
        if LLMCLIENT_OK:
            try:
                client = LLMClient()
                # Preferred Groq vision model (as specified)
                vision_models = [
                    "meta-llama/llama-4-scout-17b-16e-instruct",
                    "llama-3.2-11b-vision-preview",
                ]
                last_exc = None
                # Build multimodal messages using image_url with data URI
                messages_base = [
                    {"role": "system", "content": f"You are a concise, objective assistant. Describe the scene, actions, objects and observable behaviors in the provided images. Respond in {language}."}
                ]
                img_messages = []
                for b64 in imgs_b64:
                    img_messages.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    })
                user_content = img_messages + [{"type": "text", "text": "Provide a concise 2-4 sentence summary and list notable actions/objects."}]
                messages = messages_base + [{"role": "user", "content": user_content}]

                for vm in vision_models:
                    try:
                        client.model = vm
                        desc = client.generate_from_messages(messages, max_tokens=512)
                        return {
                            "description": desc.strip(),
                            "method_used": f"groq-vision-fallback:{vm}",
                            "frames_analyzed": len(frames),
                            "success": True,
                        }
                    except Exception as e:
                        last_exc = e
                        logger.warning(f"Groq model {vm} failed: {e}")
                if last_exc:
                    raise last_exc
            except Exception as e:
                logger.warning(f"Groq fallback failed: {e}")

        # Final fallback: simple concatenation of placeholder descriptions
        descriptions = [f"Frame {i+1}/{len(frames)} : scène vidéo" for i in range(len(frames))]
        combined = self._fuse_descriptions(descriptions, "", language)
        return {
            "description": combined,
            "method_used": "local-placeholder-fallback",
            "frames_analyzed": len(frames),
            "success": bool(combined and combined != "Impossible d'analyser cette vidéo."),
        }

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
                if w > 320:
                    scale = 320 / w
                    frame = cv2.resize(frame, (320, int(h * scale)))
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
        Audio intentionally ignored — video is processed without audio.
        This function always returns an empty string.
        """
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