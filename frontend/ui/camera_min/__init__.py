import os
import streamlit.components.v1 as components

# Folder of this component
_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Verify HTML exists
if not os.path.isfile(os.path.join(_COMPONENT_DIR, "index.html")):
    raise FileNotFoundError(f"[camera_min_local] index.html not found in {_COMPONENT_DIR}")

# Register Streamlit component (MUST match the iframe name)
camera_min_component = components.declare_component(
    "camera_min_local",
    path=_COMPONENT_DIR,
)

def camera_min(key=None):
    """Simple wrapper for the component"""
    return camera_min_component(key=key, default=None)
