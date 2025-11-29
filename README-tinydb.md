# 🚀 DrawMeMaybe – Backend Setup & Deployment Guide

This document explains how to install, run, and deploy the FastAPI backend for DrawMeMaybe.
The backend handles:

creating anonymous sessions

storing consent + participant name

saving all entries in a local TinyDB (db.json)

providing simple API endpoints for the Vue frontend

📁 Project Structure (relevant part)
DrawMeMaybe/
│
├─ backend/
│   ├─ main.py
│   ├─ db.py
│   ├─ __init__.py
│
├─ frontend/
│   └─ (Vue 3 + Vite project)
│
└─ venv/                # virtual environment (created later)

⚙️ 1. Backend Installation (local development)

These steps are used on the main development PC.

1.1. Create & activate the virtual environment
Windows (PowerShell)
cd C:\Users\<USER>\drawmemaybe\DrawMeMaybe
python -m venv venv
.\venv\Scripts\activate

Linux
cd ~/drawmemaybe/DrawMeMaybe
python3 -m venv venv
source venv/bin/activate

1.2. Install dependencies

Inside the activated venv:

pip install fastapi "uvicorn[standard]" tinydb

📡 2. FastAPI Application

The backend exposes three main endpoints:

Method    Path        Description
GET       /health     Health check
POST      /session    Create a new session
POST      /consent    Store user consent + name

The backend files:

2.1. backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .db import create_session, save_consent

app = FastAPI()

# Allow calls from the development frontend

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
    name: str | None = None

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

2.2. backend/db.py
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

def save_consent(session_id: str, consent_given: bool, name: str | None = None):
    entry = {
        "session_id": session_id,
        "consent_given": consent_given,
        "timestamp": now_iso(),
    }
    if name is not None:
        entry["name"] = name

    consents_table.insert(entry)

def get_session(session_id: str):
    Session = Query()
    return sessions_table.get(Session.session_id == session_id)

▶️ 3. Running the Backend (local dev)

Start FastAPI using uvicorn:

uvicorn backend.main:app --reload

Backend runs at: <http://127.0.0.1:8000>

Useful URLs:

[http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (Swagger UI)

Keep this terminal open while developing.

🎨 4. Connecting the Frontend

In the frontend’s ConsentView:

const API_BASE = "<http://127.0.0.1:8000>";

During the workflow:

Create new session:
POST /session → returns UUID

Send consent:
POST /consent → stores name + consent flag

Entries appear in db.json

When deployed to the server, this URL will change (see below).

🚀 5. Deployment to Linux Server (no Docker!)

These steps deploy the backend permanently so tablets/clients can reach it.

5.1. Upload project to server

SSH into your Linux server:

cd ~
git clone [YOUR_REPO_URL] drawmemaybe
cd drawmemaybe/DrawMeMaybe

Or upload via scp/rsync.

5.2. Create venv + install requirements
python3 -m venv venv
source venv/bin/activate
pip install fastapi "uvicorn[standard]" tinydb

5.3. Test the backend on the server
uvicorn backend.main:app --host 0.0.0.0 --port 8000

Visit from another device:

<http://SERVER-IP:8000/health>

If it works → proceed.

Stop with Ctrl+C.

🛠️ 6. Run FastAPI as a systemd Service

This makes uvicorn start automatically on boot and restart on crash.
No Docker needed.

6.1. Create the service file
sudo nano /etc/systemd/system/drawmemaybe.service

Paste:

[Unit]
Description=DrawMeMaybe FastAPI Backend
After=network.target

[Service]
User=YOUR_USERNAME
Group=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/drawmemaybe/DrawMeMaybe
ExecStart=/home/YOUR_USERNAME/drawmemaybe/DrawMeMaybe/venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target

Replace YOUR_USERNAME and paths accordingly.

6.2. Enable & start it
sudo systemctl daemon-reload
sudo systemctl enable drawmemaybe
sudo systemctl start drawmemaybe

Check:

sudo systemctl status drawmemaybe

View logs:

sudo journalctl -u drawmemaybe -f

Your backend is now live permanently.

🌍 7. Point the Frontend to the Server

Change the backend base URL in the frontend:

const API_BASE = "[http://SERVER-IP:8000](http://SERVER-IP:8000)";

Example:

const API_BASE = "[http://192.168.1.42:8000](http://192.168.1.42:8000)";
Now tablets/clients can communicate with the backend over the network.
