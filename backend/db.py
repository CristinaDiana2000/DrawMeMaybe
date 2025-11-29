from tinydb import TinyDB, Query
from datetime import datetime
import uuid

db = TinyDB("db.json")
sessions_table = db.table("sessions")
consents_table = db.table("consents")


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def create_session() -> str:
    session_id = str(uuid.uuid4())
    sessions_table.insert(
        {
            "session_id": session_id,
            "created_at": now_iso(),
            "status": "created",
        }
    )
    return session_id


def save_consent(session_id: str, consent_given: bool):
    consents_table.insert(
        {
            "session_id": session_id,
            "consent_given": consent_given,
            "timestamp": now_iso(),
        }
    )


def get_session(session_id: str):
    Session = Query()
    return sessions_table.get(Session.session_id == session_id)
