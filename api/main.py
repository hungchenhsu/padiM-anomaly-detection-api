# === file: api/main.py === 
"""FastAPI inference server with memory-aware lazy loading."""

from __future__ import annotations

from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from torchvision import transforms as T

from model_utils import infer_once

# ------------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------------

WEIGHTS_DIR = Path("weights")               # adapt if needed
VALID_CATS = {p.stem for p in WEIGHTS_DIR.glob("*.pth")}

app = FastAPI(title="PaDiM / FastFlow Anomaly-Detection API")

# Simple preprocessing: resize to 512×512 then convert to tensor
PREPROC = T.Compose([
    T.ToTensor(),
    T.Resize(512, antialias=True),
])


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/ping")
async def ping():
    return {"msg": "pong"}


@app.post("/predict/{category}")
async def predict(category: str, file: UploadFile = File(...)):
    if category not in VALID_CATS:
        raise HTTPException(status_code=404, detail=f"Unknown category: {category}")

    # Read and decode the uploaded image
    img_bytes = await file.read()
    img_bgr = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Convert BGR → RGB → tensor
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = PREPROC(img_pil).unsqueeze(0)  # add batch dim

    # Inference (auto-quantise / auto-unload handled inside)
    try:
        score, thr = infer_once(category, img_tensor)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    label = "defect" if score > thr else "good"
    return {
        "label": label,
        "score": round(score, 4),
        "threshold": round(thr, 4),
    }