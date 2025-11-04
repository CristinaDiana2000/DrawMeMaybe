import streamlit as st

def show_consent_form():
    st.title("🧾 Einverständniserklärung")
    st.write(
        "Bitte bestätige, dass dein Foto und Text ausschließlich zur Erstellung "
        "einer personalisierten Zeichnung mit KI verarbeitet werden."
    )
    ok = st.checkbox("Ich stimme der Datenverarbeitung zu.")

    col1, col2 = st.columns([1,1])
    with col1:
        if ok and st.button("Akzeptieren & Fortfahren ✅", use_container_width=True):
            st.session_state.consent_accepted = True
            st.rerun()

    # ❗ Wichtig: solange nicht akzeptiert, hier stoppen:
    st.stop()
