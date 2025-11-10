import streamlit as st
from ui_screensaver import show_screensaver
from ui_consent import show_consent_form
from ui_camera import show_camera_stage
from ui_chat import show_chat_ui
from utils.styles import inject_global_css

st.set_page_config(page_title="Rob Ross Chat", page_icon="🎨", layout="wide")
inject_global_css()

# ---------- helpers ----------
def _read_qp() -> dict:
    try:
        qp = dict(st.query_params)
    except Exception:
        qp = st.experimental_get_query_params()
    return {k: (v if isinstance(v, str) else (v[0] if v else None)) for k, v in qp.items()}

def _set_qp(**kwargs):
    try:
        st.query_params.clear()
        for k, v in kwargs.items():
            if v is not None:
                st.query_params[k] = v
    except Exception:
        st.experimental_set_query_params(**{k: v for k, v in kwargs.items() if v is not None})

def go(route: str):
    st.session_state.route = route
    _set_qp(route=route)
    st.rerun()

# ---------- optional flags ----------
st.session_state.setdefault("consent_accepted", False)
st.session_state.setdefault("photo_captured", False)

# ---------- routing ----------
qp = _read_qp()
route_from_url = qp.get("route")  # kann None sein

# 1) Cold start: Session-Route initialisieren (URL-Route falls vorhanden, sonst screensaver)
if "route" not in st.session_state:
    st.session_state.route = route_from_url or "screensaver"

# 2) Falls URL-Route ungleich Session-Route ist, bevorzugen wir die Session
#    (z.B. nach Klick auf Accept hat die Seite st.session_state.route="camera" gesetzt)
elif route_from_url != st.session_state.route:
    # URL an die Session anpassen, damit Back/Forward sauber funktionieren
    _set_qp(route=st.session_state.route)

route = st.session_state.route

# ---------- render ----------
if route == "screensaver":
    show_screensaver()

elif route == "consent":
    show_consent_form()

elif route == "camera":
    show_camera_stage()

elif route == "chat":
    show_chat_ui()

else:
    go("screensaver")
