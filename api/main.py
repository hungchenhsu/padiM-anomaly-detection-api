# === file: api/main.py ===
"""FastAPI inference server (lazy‑load detectors)."""
from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2, numpy as np
from detector import PaDiMDetector
from config import WEIGHTS_DIR

app = FastAPI(title="PaDiM Anomaly Detection API")

_DETECTORS: dict[str, PaDiMDetector] = {}

# all valid categories = list(.pth)
_VALID_CATS = {p.stem for p in WEIGHTS_DIR.glob("*.pth")}


def _get_detector(cat: str) -> PaDiMDetector:
    if cat not in _DETECTORS:
        _DETECTORS[cat] = PaDiMDetector(cat)
    return _DETECTORS[cat]


@app.get("/ping")
async def ping():
    return {"msg": "pong"}


@app.post("/predict/{category}")
async def predict(category: str, file: UploadFile = File(...)):
    if category not in _VALID_CATS:
        raise HTTPException(404, f"Unknown category '{category}'")

    img_bytes = await file.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Invalid image file")

    det   = _get_detector(category)         # ← lazy load here
    score = det.predict(img)
    label = "defect" if score > 9.0 else "good"  # TODO: per‑category threshold
    return {"label": label, "score": round(score, 4)}