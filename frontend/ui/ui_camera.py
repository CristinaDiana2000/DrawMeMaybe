import base64, re, os
import streamlit as st
import camera_min

def show_camera_stage():
    ss = st.session_state
    ss.setdefault("photo_captured", False)
    ss.setdefault("camera_photo_bytes", None)

    st.subheader("📸 Position yourself in the frame")
    st.caption("Laptop → webcam. Phone/Tablet → front camera. Use HTTPS or localhost.")
    st.caption(f"Component dir: {os.path.dirname(camera_min.__file__)}")

    resp = camera_min.camera_min(key="camera_min_iframe")
    st.caption(f"Component value: {resp!r}")

    if isinstance(resp, dict) and resp.get("status") == "captured":
        m = re.match(r"^data:image/[^;]+;base64,(.+)$", resp.get("dataUrl",""))
        if m: ss.camera_photo_bytes = base64.b64decode(m.group(1))
    elif isinstance(resp, dict) and resp.get("status") == "retake":
        ss.camera_photo_bytes = None

    if ss.camera_photo_bytes:
        st.success("✅ Photo captured! Review below.")
        st.image(ss.camera_photo_bytes, caption="Captured image", use_column_width=True)

    c1, c2 = st.columns(2)
    with c1:
        cont = st.button("Continue", key="btn_continue", disabled=ss.camera_photo_bytes is None)
    with c2:
        retake = st.button("Retake", key="btn_retake", disabled=ss.camera_photo_bytes is None)

    if cont and ss.camera_photo_bytes:
        ss.photo_captured = True
        st.rerun()
    if retake:
        ss.camera_photo_bytes = None
        st.rerun()
