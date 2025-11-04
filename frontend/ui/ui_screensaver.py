import streamlit as st

def show_screensaver():
    st.markdown("""
    <style>
      .screensaver {
        position: fixed; inset: 0;
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        background: #f5e3c3;
        z-index: 9998;  /* below the link overlay */
        -webkit-user-select: none; user-select: none;
        overflow: hidden;
      }
      .logo {
        width: min(60vw, 380px);
        animation: float 3s ease-in-out infinite;
      }
      .hint { margin-top: 1rem; color: #333; font-weight: 600; }
      @keyframes float {
        0% { transform: translateY(0); }
        50% { transform: translateY(-12px); }
        100% { transform: translateY(0); }
      }

      /* Full-screen clickable link */
      .tap-link {
        position: fixed; inset: 0;
        display: block;
        z-index: 10000;
        background: rgba(0,0,0,0); /* transparent, but clickable */
        cursor: pointer;
        text-decoration: none;
      }

      /* Make sure nothing else eats the click */
      header, footer, [data-testid="stToolbar"] { pointer-events: none !important; }
    </style>

    <div class="screensaver">
      <img class="logo"
           src="https://raw.githubusercontent.com/Cristina2000-hub/DrawMeMaybe/frontend/frontend/uploads/Designer%20(1).png"
           alt="logo" />
      <div class="hint">👆 Tap anywhere to start</div>
    </div>

    <!-- Full-screen anchor: navigates inside the iframe to ?touched=1 -->
    <a class="tap-link" href="?touched=1" aria-label="Start"></a>
    """, unsafe_allow_html=True)

    # Visible fallback if someone disables links somehow
    if st.button("Start 🎨", use_container_width=True):
        st.session_state.touched = True
        st.rerun()
