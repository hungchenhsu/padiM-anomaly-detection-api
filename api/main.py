# === file: api/main.py ===
"""FastAPI inference server."""
from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2, numpy as np
from detector import PaDiMDetector
from config import WEIGHTS_DIR

app = FastAPI(title="PaDiM Anomaly Detection API")

# preload detectors
_DETECTORS = {p.stem: PaDiMDetector(p.stem) for p in WEIGHTS_DIR.glob("*.pth")}

@app.post("/predict/{category}")
async def predict(category: str, file: UploadFile = File(...)):
    if category not in _DETECTORS:
        raise HTTPException(404, f"Unknown category '{category}'")
    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image")

    score = _DETECTORS[category].predict(img)
    label = "defect" if score > 9.0 else "good"  # TODO: tune threshold per category
    return {"label": label, "score": round(score, 4)}