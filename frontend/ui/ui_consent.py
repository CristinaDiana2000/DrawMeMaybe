# ui_consent.py
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
import streamlit as st
# from tinydb import TinyDB  # <- enable when ready
# db = TinyDB("consent_db.json")

TZ = ZoneInfo("Europe/Berlin")

CONSENT_TEXT_MD = r"""
### 1. Purpose of Participation
As part of a university project, your photo and a short personal input (e.g., a hobby or interest) will be used to create a personalized **caricature-style image**.  
The final drawing will be produced by a robotic arm.

### 2. Data Collected
- A photo of your face (captured during participation)  
- A hobby or interest you provide in text form  

No contact details or sensitive personal data are collected unless you provide them voluntarily.

### 3. Use of AI Systems and External Services
To generate the final caricature, the project may use:
- a text processing service to help transform your written input into a creative prompt, and  
- an image stylization service that converts your photo into a caricature-style image.

Your photo is only sent to services that are necessary for creating the caricature image.  
No personal identifiers (such as your full name or contact details) are included with the image or prompt.

### 4. Data Storage and Deletion
- Your photo and the generated caricature are used only for processing and drawing.  
- They are not stored permanently in any database or system.  
- Temporary files created during processing are deleted after the drawing is completed.  
- An anonymized record of your consent decision may be kept for documentation.

### 5. Access to Data
Access is limited to members of the project team.  
No data is shared with parties other than the AI services necessary for processing as described above.

### 6. Voluntary Participation and Right to Withdraw
Participation is voluntary.  
You may withdraw your consent at any time. If you withdraw, your data will be deleted and no caricature will be produced.  
Non-participation will not result in any disadvantage.

### 7. Contact
**Project:** DrawMeMaybe  
**Email:** <span class="no-link">fi-ws25-drawmemaybe@hs-augsburg.de</span>

### 8. Confirmation of Consent
By selecting **Accept**, you confirm that you have read and understood the above and agree to the processing of your data as described.
"""

# -------------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------------
def _ensure_session_bootstrap():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "session_start" not in st.session_state:
        st.session_state.session_start = datetime.now(TZ).isoformat()

def _now_iso():
    return datetime.now(TZ).isoformat()

def _save_consent_to_db(payload: dict):
    # db.insert(payload)
    pass

def _rerun():
    """Compatibility rerun for all Streamlit versions."""
    rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(rerun_fn):
        rerun_fn()
    else:
        # Best-effort fallback if no programmatic rerun API is available
        st.stop()

# -------------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------------
def show_consent_form():
    _ensure_session_bootstrap()

    st.markdown("""
    <style>
      /* --- Layout adjustments --- */
      [data-testid="stAppViewContainer"] .main .block-container {
          padding-top: 0.6rem !important;
          padding-bottom: 1.0rem !important;
      }

      .consent-enter { animation: fadeInUp .25s ease both; }
      @keyframes fadeInUp {
        0% {opacity: 0; transform: translateY(6px);}
        100% {opacity: 1; transform: translateY(0);}
      }

      .consent-box {
          max-height: 55vh;
          overflow: auto;
          margin-top: 0.2rem;
          margin-bottom: 0.8rem;
          background: transparent;
      }

      /* --- Text Input appearance fixes --- */

      /* Entfernt dunkle Wrapper-Hintergründe (verhindert "Halbmonde") */
      [data-testid="stTextInput"] > div,
      [data-testid="stTextInput"] > div > div {
          background: transparent !important;
          border: none !important;
          box-shadow: none !important;
          overflow: visible !important;
          border-radius: 0 !important;
      }

      /* Das eigentliche Input bleibt weiß */
      [data-testid="stTextInput"] input {
          background: #ffffff !important;
          color: #111 !important;
          border: 2px solid orange !important;
          border-radius: 9999px !important;  /* pill form */
          padding: 0.4rem 0.8rem !important;
          box-shadow: none !important;
          outline: none !important;
          -webkit-appearance: none;
                  appearance: none;
      }
      [data-testid="stTextInput"] input:focus,
      [data-testid="stTextInput"] input:active {
          background: #ffffff !important;
          color: #111 !important;
          border: 2px solid #f5b400 !important;
          box-shadow: none !important;
          outline: none !important;
      }

      /* "Press Enter to apply" Text wirklich schwarz in allen Streamlit-Versionen */
        [data-testid="stTextInput"] *,
        [data-testid="stTextInput"] :where(p, small, span, label) {
          color: #000 !important;
        }

      /* Email-Link neutral */
      .no-link { color: inherit !important; text-decoration: none !important; pointer-events: none !important; }
      .no-link a, .no-link a:link, .no-link a:visited {
          color: inherit !important; text-decoration: none !important; pointer-events: none !important;
      }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="consent-enter">', unsafe_allow_html=True)
    st.title("🧾 Consent to the Processing of Image and Personal Input for AI-Based Caricature Generation")

    # main consent text
    st.markdown('<div class="consent-box">', unsafe_allow_html=True)
    st.markdown(CONSENT_TEXT_MD, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # text input
    name = st.text_input(
        "Your name",
        placeholder="Enter your name",
        label_visibility="collapsed",
        help="Press Enter to apply"
    )

    # buttons
    col1, col2 = st.columns(2)
    with col1:
        accept = st.button("Accept ✅", use_container_width=True, type="primary")
    with col2:
        decline = st.button("Decline ❌", use_container_width=True)

    # logic
    if accept:
        if not name.strip():
            st.warning("Please enter your name to continue.")
            st.stop()

        payload = {
            "session_id": st.session_state.session_id,
            "session_start": st.session_state.session_start,
            "name": name.strip(),
            "decision": "accepted",
            "decision_timestamp": _now_iso(),
        }
        _save_consent_to_db(payload)

        st.session_state.consent_accepted = True
        st.session_state.route = "camera"
        _rerun()  # guaranteed rerun

    if decline:
        payload = {
            "session_id": st.session_state.session_id,
            "session_start": st.session_state.session_start,
            "name": name.strip(),
            "decision": "declined",
            "decision_timestamp": _now_iso(),
        }
        _save_consent_to_db(payload)
        st.error("You declined consent. The app will not proceed.")
        st.stop()

    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()
