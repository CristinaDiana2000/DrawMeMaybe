import streamlit as st

def show_consent_form():
    # small fade-in animation for the consent container
    st.markdown("""
    <style>
      .consent-enter {
        animation: fadeInUp .35s ease both;
      }
      @keyframes fadeInUp {
        0% { opacity: 0; transform: translateY(8px); }
        100% { opacity: 1; transform: translateY(0); }
      }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="consent-enter">', unsafe_allow_html=True)

    st.title("🧾 Consent Form")
    st.write(
        "Bitte bestätige, dass dein Foto und Text **ausschließlich** zur Erstellung "
        "einer personalisierten Zeichnung mit KI verarbeitet werden."
    )
    ok = st.checkbox("Ich stimme der Datenverarbeitung zu.")

    col1, col2 = st.columns([1,1])
    with col1:
        if ok and st.button("Akzeptieren & Fortfahren ✅", use_container_width=True):
            st.session_state.consent_accepted = True
            st.rerun()
    with col2:
        st.button("Abbrechen ❌", use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # stop rendering the next stages until accepted
    st.stop()
