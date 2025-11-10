import streamlit as st

def show_screensaver():
    st.markdown("""
    <style>
      .screensaver {
        position: fixed; inset: 0;
        display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        background: #ffe8b5;
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

      .tap-link {
        position: fixed; inset: 0;
        display: block;
        z-index: 10000;
        background: rgba(0,0,0,0); /* transparent, but clickable */
        cursor: pointer;
        text-decoration: none;
      }

      header, footer, [data-testid="stToolbar"] { pointer-events: none !important; }
    </style>

    <div class="screensaver">
      <img class="logo"
           src="https://raw.githubusercontent.com/Cristina2000-hub/DrawMeMaybe/frontend/frontend/uploads/Designer%20(1).png"
           alt="logo" />
      <div class="hint">👆 Tap anywhere to start</div>
    </div>

    <!-- ✅ fixed: same look, but no new tab -->
    <a class="tap-link"
       href="?route=consent"
       target="_self"
       aria-label="Start"
       onclick="
        try {
          const u = new URL(window.location.href);   // <-- this window (the iframe)
          u.searchParams.set('touched','1');
          window.location.replace(u.toString());      // <-- same tab, same iframe
        } catch (e) {
          window.location.href = (
            window.location.href +
            (window.location.search ? '&' : '?') +
            'touched=1'
          );
        }
        return false;  // prevent default anchor navigation
      ">
    </a>

    <script>
      for (const a of document.querySelectorAll('a')) {
        a.setAttribute('target','_self');
      }
    </script>
    """, unsafe_allow_html=True)

    # optional fallback button (you can remove if not wanted)
    # if st.button("Start 🎨", use_container_width=True):
    #     st.session_state.touched = True
    #     st.rerun()
