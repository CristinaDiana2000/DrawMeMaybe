import os, uuid

def create_session(base_dir="sessions"):  #speichert Sitzungsdaten in "sessions/"-Verzeichnis standardmäßig
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(base_dir, session_id)
    os.makedirs(session_dir, exist_ok=True)
    return session_dir