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

# ---------- stage flags (optional) ----------
st.session_state.setdefault("consent_accepted", False)
st.session_state.setdefault("photo_captured", False)

# ---------- read & normalize URL ----------
qp = _read_qp()
route_from_url = qp.get("route")
touched_flag = qp.get("touched") == "1"



# If neither ?route nor ?touched is present, clear any stale route so we show screensaver
if route_from_url:                           # explicit deep-link wins
    route = route_from_url
elif "route" in st.session_state:            # in-app nav
    route = st.session_state.route
else:                                        # cold start -> screensaver
    route = "screensaver"
    st.session_state.route = route

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
