# === file: streamlit_app.py ===    
"""Streamlit dashboard for PaDiM REST API."""
import base64
import io
import os

import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "https://padim-defect-detector-api.onrender.com")

st.set_page_config(page_title="PaDiM Dashboard", layout="wide")
st.title("🛠️ PaDiM Defect Detection Dashboard")
st.markdown("Upload an image, pick a category, and view the anomaly score.")

# Sidebar controls
CATEGORY = st.sidebar.selectbox(
    "Category",
    [
        "bottle", "cable", "capsule", "carpet", "grid",
        "hazelnut", "leather", "metal_nut", "pill", "screw",
        "tile", "toothbrush", "transistor", "wood", "zipper",
    ],
)

UPLOADED = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])


def show_response(resp_json: dict) -> None:
    """Render JSON plus optional heat-map."""
    st.subheader("Prediction Result")
    st.json(resp_json)

    if "heatmap" in resp_json:
        col1, col2 = st.columns(2)
        col1.image(UPLOADED, caption="Original", use_column_width=True)

        heat_bytes = base64.b64decode(resp_json["heatmap"])
        heat_img = Image.open(io.BytesIO(heat_bytes))
        col2.image(heat_img, caption="Heat-map", use_column_width=True)


if UPLOADED and st.button("Predict"):
    files = {"file": (UPLOADED.name, UPLOADED.getvalue(), UPLOADED.type)}
    with st.spinner("Calling API…"):
        try:
            resp = requests.post(
                f"{API_URL}/predict/{CATEGORY}",
                files=files,
                timeout=180,          # increase to avoid ReadTimeout
            )
        except requests.exceptions.RequestException as exc:
            st.error(f"Request failed: {exc}")
        else:
            if resp.ok:
                show_response(resp.json())
            else:
                st.error(resp.text)