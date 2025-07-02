# === file: streamlit_app.py ===
"""Streamlit dashboard for PaDiM REST API."""
import streamlit as st, requests, cv2, numpy as np, base64
from PIL import Image

import os
API_URL = os.getenv("API_URL", "https://padim-defect-detector-api.onrender.com")

st.set_page_config(page_title="PaDiM Dashboard", layout="wide")
st.title("🛠️ PaDiM Defect Detection Dashboard")
st.markdown("Upload an image, pick a category, and view the anomaly score.")

# Sidebar controls
category = st.sidebar.selectbox("Category", [
    "bottle","cable","capsule","carpet","grid","hazelnut","leather",
    "metal_nut","pill","screw","tile","toothbrush","transistor","wood","zipper"
])

uploaded = st.file_uploader("Choose an image", type=["png","jpg","jpeg"])

def show_response(resp_json):
    st.subheader("Prediction Result")
    st.json(resp_json)
    if "heatmap" in resp_json:  # if API returns heatmap b64
        col1, col2 = st.columns(2)
        col1.image(uploaded, caption="Original", use_column_width=True)
        heat_bytes = base64.b64decode(resp_json["heatmap"])
        heat_img = Image.open(io.BytesIO(heat_bytes))
        col2.image(heat_img, caption="Heatmap", use_column_width=True)

if uploaded and st.button("Predict"):
    files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
    data = {"category": category}
    with st.spinner("Calling API..."):
        r = requests.post(f"{API_URL}/predict/{category}", files=files)
        if r.status_code == 200:
            show_response(r.json())
        else:
            st.error(f"API Error {r.status_code}: {r.text}")