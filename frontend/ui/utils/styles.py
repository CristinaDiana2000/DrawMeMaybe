import streamlit as st

def inject_global_css():
    st.markdown("""
<style>
/* Full gradient background */
html, body, .stApp,
body > div, body > div > div {
    margin: 0;
    padding: 0;
    height: 100%;
    font-family: "Poppins", sans-serif;
    color: #000;
    background-color: #f5e3c3;
    background-image:
        radial-gradient(circle at 20% 30%, #ff6f61 0%, transparent 40%),
        radial-gradient(circle at 80% 20%, #00bcd4 0%, transparent 40%),
        radial-gradient(circle at 30% 80%, #8bc34a 0%, transparent 40%),
        radial-gradient(circle at 70% 70%, #ffeb3b 0%, transparent 40%),
        radial-gradient(circle at 50% 50%, #e91e63 0%, transparent 40%);
    background-repeat: no-repeat;
    background-size: cover;
    background-attachment: fixed;
    overflow: hidden;
}

/* Remove dark backgrounds */
header[data-testid="stHeader"],
footer,
body > div:last-child,
body > div[data-testid="stStatusWidget"],
div[data-testid="stBottomBlockContainer"],
div[data-testid="stChatInput"],
div[class*="st-emotion-cache"][class*="e4man"],

div[class*="st-emotion-cache"][class*="e196pkbe"],
div[class*="st-emotion-cache"][class*="e1gk92lc"],
div[class*="st-emotion-cache"][class*="ex0cdmw"] {
    background: transparent !important;
    box-shadow: none !important;
}

/* Layout */
.stApp {
    display: flex;
    flex-direction: column;
    height: 100vh;
}

.block-container {
    flex: 1;
    max-width: 650px;
    margin: 0 auto;
    padding: 2rem 0.5rem 0 0.5rem;
    display: flex;
    flex-direction: column;
}

/* Scrollable chat area */
.chat-scroll {
    flex: 1;
    overflow-y: auto;
    padding-right: 0.5rem;
    z-index: 1;
}

/* Title */
h1 {
    text-align: center;
    color: #000;
    font-weight: 700;
    margin-bottom: 2rem;
}

/* Chat bubbles */
.chat-bubble {
    display: flex;
    align-items: flex-start;
    margin: 0.75rem 0;
    width: 100%;
}

.bubble {
    max-width: 80%;
    padding: 0.9rem 1.2rem;
    border-radius: 18px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    font-size: 1rem;
    line-height: 1.5;
    color: #000;
    background-color: rgba(255,255,255,0.9);
}

.chat-left {
    justify-content: flex-start;
}
.chat-left .bubble {
    background-color: #E3F7F7;
    border: 1px solid #80C6C6;
    margin-left: 0.75rem;
}
.chat-left .avatar {
    order: -1;
    margin-right: 0.5rem;
    width="64" height="64"
}

.chat-right {
    justify-content: flex-end;
}
.chat-right .bubble {
    background-color: #FFF3E0;
    border: 1px solid #FFC470;
    margin-right: 0.75rem;
}
.chat-right .avatar {
    order: 2;
    margin-left: 0.5rem;
}

.avatar {
    width: 64px;
    height: 64px;
    flex-shrink: 0;
}

/* Input container */
.stChatInputContainer {
    padding: 0.5rem 0;
    background: transparent;
    position: relative;
    z-index: 2;
    display: flex;
    justify-content: center;
}

/* Frosted input wrapper -> solid white fill for clear, readable input */
div[data-testid="stChatInput"] {
    backdrop-filter: none;
    background-color: #ffffff !important;
    border-radius: 24px;
    padding: 0.35rem 0.6rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    max-width: 380px !important;
    margin: 0 auto;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Text input field */
div[data-baseweb="input"] {
    width: 240px !important;
    height: 44px !important;
    margin: 0 auto !important;
    padding: 0.2rem 0.9rem !important;
    border: 2px solid #FFA500 !important;
    border-radius: 999px !important;
    background-color: #ffffff !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    display: flex;
    align-items: center;
}

/* Input text styling */
div[data-baseweb="input"] input,
div[data-baseweb="input"] textarea,
div[data-baseweb="input"] [contenteditable="true"],
div[data-baseweb="input"] div[role="combobox"] {
    color: #000 !important;
    font-size: 0.95rem;
    font-weight: 500;
    background-color: #ffffff !important;
    border: none !important;
    outline: none !important;
    width: 100%;
}

/* Base-input and chat textarea */
div[data-baseweb="base-input"] textarea,
textarea[data-testid="stChatInputTextArea"] {
    background-color: #ffffff !important;
    color: #000 !important;
    border: none !important;
    outline: none !important;
    resize: none !important;
    width: 100%;
}

/* Remove orange focus */
div[data-baseweb="input"] input:focus,
div[data-baseweb="input"] textarea:focus,
div[data-baseweb="base-input"] textarea:focus,
textarea[data-testid="stChatInputTextArea"]:focus,
div[data-testid="stChatInput"] [contenteditable="true"]:focus,
div[data-baseweb="input"]:focus,
div[data-baseweb="base-input"]:focus,
div[data-testid="stChatInput"] :focus {
    outline: none !important;
    box-shadow: none !important;
    border-color: #ffffff !important;
}

/* Focus-visible ring */
div[data-testid="stChatInput"] textarea:focus-visible,
div[data-testid="stChatInput"] input:focus-visible,
div[data-baseweb="input"] input:focus-visible,
textarea[data-testid="stChatInputTextArea"]:focus-visible,
div[data-baseweb="base-input"] textarea:focus-visible {
    outline: 2px solid rgba(0,0,0,0.85) !important;
    box-shadow: 0 0 0 4px rgba(0,0,0,0.06) !important;
    border-color: rgba(0,0,0,0.85) !important;
}

/* Make wrappers white too */
div[data-testid="stChatInput"] div[data-baseweb="textarea"],
div[data-testid="stChatInput"] div[data-baseweb="base-input"],
div[data-testid="stChatInput"] div[data-baseweb="input"],
div[data-testid="stChatInput"] [class^="st-emotion-cache"],
div[data-testid="stChatInput"] [class*="st-emotion-cache"] {
    background-color: #ffffff !important;
    color: #000 !important;
    box-shadow: none !important;
    border-radius: 18px !important;
}

/* Placeholder */
div[data-baseweb="input"] input::placeholder {
    color: #6b6b6b !important;
    opacity: 1 !important;
}

/* Send button */
button[data-testid="stChatInputSubmitButton"] {
    background-color: #FFA500 !important;
    border: none !important;
    border-radius: 50%;
    padding: 0.4rem;
    color: #ffffff !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}
</style>
    """, unsafe_allow_html=True)
