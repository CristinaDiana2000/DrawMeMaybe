import streamlit as st

st.set_page_config(
    page_title="Rob Ross Chat",
    page_icon="🎨",
    layout="wide"
)

# ---------------------- GLOBAL CSS ----------------------
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
    /* keep rounded look but make it solid white so the input appears filled */
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

/* Text input field: compact, oval, filled white with clear black text */
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
/* Ensure the actual editable/input element is solid white and black text.
   Some Streamlit/baseweb structures use input, textarea, or contenteditable divs;
   target the likely variations and force a white background so there isn't a
   dark inner box inside the white wrapper. */
div[data-baseweb="input"] input,
div[data-baseweb="input"] textarea,
div[data-baseweb="input"] [contenteditable="true"],
div[data-baseweb="input"] div[role="combobox"] {
    color: #000 !important;
    font-size: 0.95rem;
    font-weight: 500;
    background-color: #ffffff !important; /* force white fill */
    border: none !important;
    outline: none !important;
    width: 100%;
}

/* Also cover Streamlit/baseweb's base-input and the specific chat textarea
   Streamlit sometimes uses data-baseweb="base-input" and the textarea
   has data-testid="stChatInputTextArea" — target those to avoid a dark
   inner box showing through. */
div[data-baseweb="base-input"] textarea,
textarea[data-testid="stChatInputTextArea"] {
    background-color: #ffffff !important;
    color: #000 !important;
    border: none !important;
    outline: none !important;
    resize: none !important;
    width: 100%;
}

/* Make Streamlit/baseweb wrapper elements inside the chat input white too
   - targets data-baseweb="textarea" and generic emotion-cache wrappers
   - scoped under the chat input container to avoid global changes */
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

/* Placeholder color for readability */
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

# ---------------------- ICONS ----------------------
USER_ICON = "https://cdn-icons-png.flaticon.com/512/1077/1077012.png"
AI_ICON = "https://raw.githubusercontent.com/Cristina2000-hub/DrawMeMaybe/frontend/frontend/uploads/Designer%20(1).png"



# ---------------------- HEADER ----------------------
st.title("🎨 Rob Ross — Chat")

# ---------------------- CHAT STATE ----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------- CHAT DISPLAY ----------------------
st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)
for msg in st.session_state.messages:
    if msg["role"] == "assistant":
        st.markdown(f"""
        <div class="chat-bubble chat-left">
            <img src="{AI_ICON}" class="avatar" >
            <div class="bubble">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-bubble chat-right">
            <div class="bubble">{msg["content"]}</div>
            <img src="{USER_ICON}" class="avatar">
        </div>
        """, unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------------- USER INPUT ----------------------
prompt = st.chat_input("Tell me about your hobbies or interests...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    st.markdown(f"""
        <div class="chat-bubble chat-right">
            <div class="bubble">{prompt}</div>
            <img src="{USER_ICON}" class="avatar">
        </div>
    """, unsafe_allow_html=True)

    emoji_map = {
        "art": "🎨", "music": "🎵", "gardening": "🌱",
        "gaming": "🎮", "coding": "💻", "reading": "📚",
        "travel": "✈️", "sports": "⚽", "cooking": "🍳"
    }

    keyword = prompt.lower().strip()
    emoji = emoji_map.get(keyword, "✨")
    ai_response = f'Got it! You mentioned **{keyword}** — that sounds inspiring! {emoji}'

    st.session_state.messages.append({"role": "assistant", "content": ai_response})

    st.markdown(f"""
        <div class="chat-bubble chat-left">
            <img src="{AI_ICON}" class="avatar">
            <div class="bubble">{ai_response}</div>
        </div>
    """, unsafe_allow_html=True)
