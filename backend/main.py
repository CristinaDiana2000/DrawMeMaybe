from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import base64

from .tinydb.db import create_session, save_consent, list_sessions, list_consents, complete_session

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

class ChatPayload(BaseModel):
    user_text: str
    
class PhotoPayload(BaseModel):
    session_id: str
    data_url: str

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

@app.post("/api/chat")
async def chat_endpoint (payload: ChatPayload):
    # Placeholder endpoint for chat functionality
    hobbies = [payload.user_text.strip()]
    return {"success": True, "hobbies": hobbies, "prompt": "", "exaggerations": []}

@app.post("/photo")
async def upload_photo(payload: PhotoPayload):
    header, _, b64data = payload.data_url.partition(",")
    img_bytes = base64.b64decode(b64data)

    folder = Path("backend/ai_image/sd_pipeline/assets")
    folder.mkdir(parents=True, exist_ok=True)
    img_path = folder / f"{payload.session_id}.png"
    img_path.write_bytes(img_bytes)

    return {"ok": True, "path": str(img_path)}