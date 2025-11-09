import os
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.dirname(os.path.abspath(__file__))

camera_min_component = components.declare_component(
    "camera_min_local",
    path=_COMPONENT_DIR
)

def camera_min(key=None):
    return camera_min_component(key=key, default=None)
