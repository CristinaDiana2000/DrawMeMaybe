import streamlit as st

USER_ICON = "https://cdn-icons-png.flaticon.com/512/1077/1077012.png"
AI_ICON = "https://raw.githubusercontent.com/Cristina2000-hub/DrawMeMaybe/frontend/frontend/uploads/Designer%20(1).png"

def show_chat_ui():
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
            content_html = msg["content"]
            if "hobby" in st.session_state and st.session_state.hobby == msg["content"]:
                content_html = f"<strong>{msg['content']}</strong>"

            st.markdown(f"""
            <div class="chat-bubble chat-right">
                <div class="bubble">{content_html}</div>
                <img src="{USER_ICON}" class="avatar">
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------------- USER INPUT ----------------------
    prompt = st.chat_input("Tell me about your hobbies or interests...")

    if prompt:
        st.session_state.hobby = prompt
        st.session_state.messages.append({"role": "user", "content": prompt})

        st.markdown(f"""
            <div class="chat-bubble chat-right">
                <div class="bubble"><strong>{prompt}</strong></div>
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
