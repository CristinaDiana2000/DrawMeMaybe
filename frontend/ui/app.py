import streamlit as st
from ui_screensaver import show_screensaver
from ui_consent import show_consent_form
from ui_chat import show_chat_ui
from utils.styles import inject_global_css

st.set_page_config(page_title="Rob Ross Chat", page_icon="🎨", layout="wide")

# 👉 inject your (unchanged) CSS once
inject_global_css()

# --- read ?touched=1 from URL and update state ---
try:
    # Newer Streamlit
    qp = st.query_params
    if "touched" in qp and qp["touched"] == "1":
        st.session_state.touched = True
        st.query_params.clear()   # remove param
        st.rerun()
except Exception:
    # Fallback for older versions
    qp = st.experimental_get_query_params()
    if qp.get("touched", ["0"])[0] == "1":
        st.session_state.touched = True
        st.experimental_set_query_params()  # clear params
        st.rerun()


# 👉 initialize state used for flow control
if "touched" not in st.session_state:
    # set to True to skip screensaver while developing
    st.session_state.touched = False
if "consent_accepted" not in st.session_state:
    # set to True to skip consent while developing
    st.session_state.consent_accepted = False

# 👉 flow
if not st.session_state.touched:
    show_screensaver()
elif not st.session_state.consent_accepted:
    show_consent_form()
else:
    show_chat_ui()
