from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .db import create_session, save_consent

app = FastAPI()

# Allow your Vue dev server to talk to this API
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConsentPayload(BaseModel):
    session_id: str
    consent_given: bool


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
    )
    return {"ok": True}
