# === file: detector.py ===
"""Runtime detector with **lazy‑load** and **shared backbone** to reduce memory."""
import torch, cv2, numpy as np, pathlib
from config import DEVICE, WEIGHTS_DIR
from model_utils import get_backbone, FeatureExtractor
import json, os
THR_DB = {}
if os.path.exists("optimal_thresholds.json"):
    with open("optimal_thresholds.json") as f:
        THR_DB = json.load(f)

# ── shared backbone cache: {arch: torch.nn.Module} ──
_BACKBONES: dict[str, torch.nn.Module] = {}


def _get_shared_backbone(arch: str) -> torch.nn.Module:
    """Return a backbone; create & cache if not yet loaded."""
    if arch not in _BACKBONES:
        model, _ = get_backbone("grid" if arch.startswith("wide") else "bottle")
        _BACKBONES[arch] = model.to(DEVICE)
    return _BACKBONES[arch]


class PaDiMDetector:
    """Load per‑category Gaussian stats; reuse shared backbone."""
    def __init__(self, category: str):
        state = torch.load(WEIGHTS_DIR / f"{category}.pth", map_location=DEVICE)
        arch   = state["arch"]              # 'resnet34' or 'wide_resnet50_2'
        layers = state["layers"]

        backbone = _get_shared_backbone(arch)
        self.extractor = FeatureExtractor(backbone, layers)
        self.mean      = state["mean"].to(DEVICE)
        self.cov_inv   = state["cov_inv"].to(DEVICE)

        self.thr = THR_DB.get(category, 9.0)  # Default threshold if not found

    @torch.no_grad()
    def predict(self, img_bgr: np.ndarray) -> float:
        """Return image‑level anomaly score (float)."""
        from torchvision import transforms as T
        tf = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tensor = tf(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(DEVICE)
        fmap = self.extractor(tensor)                    # [1,C,H,W]
        C = fmap.shape[1]
        flat = fmap.permute(0, 2, 3, 1).reshape(-1, C)
        diff = flat - self.mean
        dist = (diff @ self.cov_inv * diff).sum(1)
        return dist.max().item()