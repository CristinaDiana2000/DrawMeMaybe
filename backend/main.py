import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathlib import Path
from PIL import Image
from backend.ai_image.agent.caricature_agent import run_caricature_agent
#from .tinydb.db import create_session, save_consent, list_sessions, list_consents, complete_session

app = FastAPI()

# Allow  Vue dev server to talk to this API
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    #allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConsentPayload(BaseModel):
    session_id: str
    consent_given: bool
    name: str | None = None

class SessionCompletePayload(BaseModel):
    session_id: str

class ChatRequest(BaseModel):
    user_text: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/session")
def create_session_endpoint():
    session_id = create_session()
    return {"session_id": session_id}


@app.post("/consent")
def consent_endpoint(payload: ConsentPayload):
    save_consent(
        session_id=payload.session_id,
        consent_given=payload.consent_given,
        name=payload.name,
    )
    return {"ok": True}

@app.post("/session/complete")
def complete_session_endpoint(payload: SessionCompletePayload):
    """
    Markiert eine Session als abgeschlossen.
    Wird normalerweise vom letzten Screen (z.B. Danke-Screen) aufgerufen.
    """
    success = complete_session(payload.session_id)
    if not success:
        # Session nicht gefunden
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


@app.get("/debug/sessions")
def debug_sessions():
    """
    Debug endpoint: returns all sessions and consents from TinyDB.
    """
    sessions = list_sessions()
    consents = list_consents()
    return {
        "sessions": sessions,
        "consents": consents,
    }
<<<<<<< HEAD
=======

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        # Same logic as run_agent_only.py
        PROJECT_ROOT = Path(__file__).resolve().parent.parent  # backend → DrawMeMaybe
        IMAGE_PATH = PROJECT_ROOT / "backend" / "ai_image" / "sd_pipeline" / "assets" / "whiteguy2.png"
        
        image = Image.open(IMAGE_PATH).convert("RGB").resize((512, 512))
        result = run_caricature_agent(image, request.user_text)
        json_path = PROJECT_ROOT / "backend" / "ai_image" / "sd_pipeline" / "agent_output.json"
        with open(json_path, "w") as f:
            json.dump(result, f, indent=2)

        
        return {
            "success": True,
            "hobbies": result.get("hobbies", []),
            "prompt": result.get("prompt", ""),
            "exaggerations": result.get("exaggerations", [])
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
>>>>>>> e9d763e57 (changed chat structure so that it work with ai)
