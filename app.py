"""
=============================================================================
app.py  —  API FastAPI — Assistant Intelligent TSA & RM
=============================================================================
Endpoints :

    POST /chat          → question texte → réponse LLM
    POST /chat/media    → question + fichier media → réponse LLM
    GET  /health        → statut de l'API

Corps des requêtes envoyés par le frontend :
    {
        "question"    : "Comment gérer les crises ?",
        "profile"     : { "prediction": "TSA", "Age_Years": 5, ... },
        "conversation": {
            "last_5_messages": [...],
            "summary"        : "...",
            "keywords"       : [...],
            "total_messages" : 7
        },
        "child": {
            "id"              : "child_123",
            "profile_detecter": ["non verbal", ...]
        }
    }

Réponse retournée au frontend :
    {
        "answer"      : "réponse finale (langue parent)",
        "parent_lang" : "fr",
        "rag_score"   : 0.74,
        "web_triggered"  : false,
        "domain_blocked" : false,
        "critical_alert" : false,
        "updates": {
            "summary"         : "...",
            "keywords"        : [...],
            "profile_detecter": [...],
            "should_update_db": true
        }
    }

INSTALLATION :
    pip install fastapi uvicorn python-multipart

LANCEMENT :
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

=============================================================================
"""

import os, shutil, logging, tempfile, json
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image
import io

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Import pipeline ────────────────────────────────────────────────────────
from src.llm.main_pipeline import MainPipeline

# ── Import services ML/RM ──────────────────────────────────────────────────
from services.ml_service     import predict_ml
from services.cnn_service    import predict_cnn
from services.fusion_service import fusion_prediction
from services.rm_service     import RMService

# ── Import NLP Dashboard ───────────────────────────────────────────────────
from src.api.nlp_dashboard import router as nlp_router


PR_QN1_A_INDEX = 10

# INITIALISATION
app = FastAPI(
    title       = "Assistant Intelligent TSA & RM",
    description = "API de conseils personnalisés pour parents d'enfants TSA/RM",
    version     = "1.0.0",
)

# CORS — autoriser le frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# NLP Dashboard router
app.include_router(nlp_router)

# Initialiser le pipeline une seule fois au démarrage
pipeline  : Optional[MainPipeline] = None
rm_service: Optional[RMService]    = None

@app.on_event("startup")
async def startup():
    global pipeline, rm_service
    logger.info("Chargement du pipeline...")
    pipeline   = MainPipeline()
    rm_service = RMService()
    logger.info("Pipeline prêt ✔")


# SCHÉMAS PYDANTIC

class Message(BaseModel):
    role   : str   # "user" | "assistant"
    content: str

class Conversation(BaseModel):
    last_5_messages: list[Message] = []
    summary        : str           = ""
    keywords       : list[str]     = []
    total_messages : int           = 0

class Child(BaseModel):
    id              : str       = ""
    profile_detecter: list[str] = []

class ChatRequest(BaseModel):
    question    : str
    profile     : dict
    conversation: Conversation = Conversation()
    child       : Child        = Child()

class ChatResponse(BaseModel):
    answer         : str
    parent_lang    : str
    rag_score      : float
    web_triggered  : bool
    domain_blocked : bool
    critical_alert : bool
    updates        : dict


# SCHÉMAS PRÉDICTION

class PredictResponse(BaseModel):
    # TSA
    prob_ml          : float
    prob_cnn         : float
    prob_tsa         : float
    tsa_detected     : bool
    # RM
    score_anomalie   : float
    rm_detected      : bool
    # Résultat final
    prediction       : str    # "TSA" | "RM" | "MIXTE" | "Normal"
    confidence       : float


# ENDPOINTS
@app.get("/")
def root():
    return {"message": "AI API running 🚀"}


# ── Prédiction TSA + RM → profil final ───────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
async def predict(
    features_tsa: str        = Form(...),   # JSON list — questionnaire TSA
    features_rm : str        = Form(...),   # JSON list — questionnaire RM
    image       : UploadFile = File(...),   # photo de l'enfant
):
    """
    Endpoint de prédiction combinée TSA + RM.

    Logique de fusion :
        TSA  oui + RM non  → "TSA"
        TSA  non + RM oui  → "RM"
        TSA  oui + RM oui  → "MIXTE"
        TSA  non + RM non  → "Normal"

    Seuils :
        TSA détecté  : prob_tsa >= 0.5
        RM détecté   : score_anomalie <= 0.3 (profil RM typique)
    """
    # ── Parser les features ───────────────────────────────────────────────
    try:
        feats_tsa = json.loads(features_tsa)
        feats_rm  = json.loads(features_rm)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JSON invalide : {e}")

    # ── Étape 1 : Prédiction TSA ─────────────────────────────────────────────
    prob_ml  = predict_ml(feats_tsa)
 
    img_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    prob_cnn  = predict_cnn(pil_image)
 
    prob_tsa     = fusion_prediction(prob_ml, prob_cnn)
    tsa_detected = prob_tsa >= 0.5
 
    logger.info(
        f"TSA : prob_ml={prob_ml:.3f} | prob_cnn={prob_cnn:.3f} | "
        f"prob_tsa={prob_tsa:.3f} | tsa_detected={tsa_detected}"
    )
 
    # ── Étape 2 : INJECTION AUTOMATIQUE DE PR_QN1_A ─────────────────────────
    # La valeur est basée sur le résultat TSA détecté :
    #   tsa_detected=True  → PR_QN1_A = 2 (Oui, diagnostiqué)
    #   tsa_detected=False → PR_QN1_A = 0 (Non)
    # Cela signifie que le modèle RM reçoit une information cohérente
    # avec le résultat TSA — pas ce que le parent a saisi (qui était 0 par défaut)
 
    pr_qn1a_value = 2 if tsa_detected else 0
 
    if len(feats_rm) > PR_QN1_A_INDEX:
        feats_rm[PR_QN1_A_INDEX] = pr_qn1a_value
        logger.info(f"PR_QN1_A injecté : {pr_qn1a_value} (tsa_detected={tsa_detected})")
    else:
        logger.warning(
            f"features_rm trop court ({len(feats_rm)} éléments) — "
            f"PR_QN1_A à l'index {PR_QN1_A_INDEX} non trouvé"
        )

    # ── Prédiction RM ─────────────────────────────────────────────────────
    rm_result      = rm_service.predict(feats_rm)
    score_anomalie = rm_result["score_anomalie"]
    rm_detected    = not rm_result["is_anomaly"]

    # ── Fusion TSA + RM → prédiction finale ──────────────────────────────
    if tsa_detected and rm_detected:
        prediction = "MIXTE"
        confidence = round((prob_tsa + (1 - score_anomalie)) / 2, 4)
    elif tsa_detected:
        prediction = "TSA"
        confidence = round(prob_tsa, 4)
    elif rm_detected:
        prediction = "RM"
        confidence = round(1 - score_anomalie, 4)
    else:
        prediction = "Normal"
        confidence = round(1 - prob_tsa, 4)

    logger.info(
        f"Prédiction : {prediction} | "
        f"TSA={prob_tsa:.2f} | RM score={score_anomalie:.2f} | "
        f"confidence={confidence}"
    )

    return PredictResponse(
        prob_ml        = round(prob_ml, 4),
        prob_cnn       = round(prob_cnn, 4),
        prob_tsa       = round(prob_tsa, 4),
        tsa_detected   = tsa_detected,
        score_anomalie = round(score_anomalie, 4),
        rm_detected    = rm_detected,
        prediction     = prediction,
        confidence     = confidence,
    )


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status" : "ok",
        "pipeline": pipeline is not None,
    }


# ── Chat texte ────────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Endpoint principal — question texte → réponse LLM.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline non initialisé")

    try:
        result = pipeline.run(
            question     = request.question,
            profile      = request.profile,
            conversation = {
                "last_5_messages": [m.dict() for m in request.conversation.last_5_messages],
                "summary"        : request.conversation.summary,
                "keywords"       : request.conversation.keywords,
                "total_messages" : request.conversation.total_messages,
            },
            child = {
                "id"              : request.child.id,
                "profile_detecter": request.child.profile_detecter,
            },
        )

        return ChatResponse(**result)

    except Exception as e:
        logger.error(f"Erreur /chat : {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Chat avec média ───────────────────────────────────────────────────────────

@app.post("/chat/media", response_model=ChatResponse)
async def chat_media(
    profile     : str                    = Form(...),
    conversation: str                    = Form("{}"),
    child       : str                    = Form("{}"),
    question    : Optional[str]          = Form(None),    # texte (si pas audio)
    media       : Optional[UploadFile]   = File(None),   # image ou vidéo
    audio       : Optional[UploadFile]   = File(None),   # message vocal
):
    """
    Endpoint avec média.

    Deux modes :
        Mode 1 — Audio seul :
            audio = fichier mp3/wav/ogg
            → transcription Whisper → devient la question

        Mode 2 — Texte + (image ou vidéo optionnel) :
            question = texte du parent
            media    = image (jpg/png) ou vidéo (mp4/avi)
            → description BLIP/vidéo ajoutée au contexte

    Règle : audio OU (question + media optionnel) — pas les deux.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline non initialisé")

    # Valider : il faut au moins audio ou question
    if not audio and not question:
        raise HTTPException(
            status_code=400,
            detail="Fournir 'audio' (message vocal) ou 'question' (texte)."
        )

    import json
    try:
        profile_dict      = json.loads(profile)
        conversation_dict = json.loads(conversation)
        child_dict        = json.loads(child)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JSON invalide : {e}")

    tmp_audio = None
    tmp_media = None

    try:
        # ── Mode 1 : Audio → question ─────────────────────────────────────
        if audio:
            ext_audio = os.path.splitext(audio.filename or "")[1].lower() or ".ogg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext_audio) as f:
                shutil.copyfileobj(audio.file, f)
                tmp_audio = f.name

            result = pipeline.run(
                question     = "",           # sera extrait de l'audio
                profile      = profile_dict,
                conversation = conversation_dict,
                child        = child_dict,
                media_path   = tmp_audio,
                media_type   = "audio",
            )

        # ── Mode 2 : Texte + (image/vidéo optionnel) ─────────────────────
        else:
            media_path = ""
            media_type = ""

            if media and media.filename:
                ext = os.path.splitext(media.filename)[1].lower()
                media_type = _detect_media_type(ext)

                if not media_type or media_type == "audio":
                    raise HTTPException(
                        status_code=400,
                        detail=f"Format non supporté pour 'media' : {ext}. "
                               f"Utiliser 'audio' pour les fichiers audio."
                    )

                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
                    shutil.copyfileobj(media.file, f)
                    tmp_media = f.name
                media_path = tmp_media

            result = pipeline.run(
                question     = question,
                profile      = profile_dict,
                conversation = conversation_dict,
                child        = child_dict,
                media_path   = media_path,
                media_type   = media_type,
            )

        return ChatResponse(**result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur /chat/media : {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        for tmp in [tmp_audio, tmp_media]:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
                logger.info(f"Fichier temporaire supprimé : {tmp}")


# UTILITAIRES

def _detect_media_type(ext: str) -> str:
    """Détecte le type de média depuis l'extension du fichier."""
    images = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    videos = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}
    audios = {".mp3", ".wav", ".ogg", ".m4a", ".flac", ".opus"}

    if ext in images: return "image"
    if ext in videos: return "video"
    if ext in audios: return "audio"
    return ""


# LANCEMENT DIRECT

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)