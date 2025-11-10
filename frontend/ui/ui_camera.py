import base64, re, os
import streamlit as st
import camera_min  # ← important: just import the module, not a subpath

def show_camera_stage(on_done=None):
    st.header("📷 Camera")
    ss = st.session_state
    ss.setdefault("photo_captured", False)
    ss.setdefault("camera_photo_bytes", None)

    st.subheader("📸 Position yourself in the frame")
    st.caption("Laptop → webcam. Phone/Tablet → front camera. Use HTTPS or localhost.")
    st.caption(f"Component dir: {os.path.dirname(camera_min.__file__)}")

    # --- Render the HTML/JS component ---
    resp = camera_min.camera_min(key="camera_min_iframe")
    st.caption(f"Component value: {resp!r}")

    # --- React to messages from JS ---
    if isinstance(resp, dict):
        if resp.get("status") == "captured":
            m = re.match(r"^data:image/[^;]+;base64,(.+)$", resp.get("dataUrl",""))
            if m:
                ss.camera_photo_bytes = base64.b64decode(m.group(1))
        elif resp.get("status") == "retake":
            ss.camera_photo_bytes = None

    # --- Buttons for flow control ---
    col1, col2 = st.columns(2)
    with col1:
        cont = st.button("Continue", disabled=not ss.camera_photo_bytes)
    with col2:
        retake = st.button("Retake", disabled=not ss.camera_photo_bytes)

    if cont and ss.camera_photo_bytes:
        ss.photo_captured = True
        st.rerun()
    if retake:
        ss.camera_photo_bytes = None
        st.rerun()
