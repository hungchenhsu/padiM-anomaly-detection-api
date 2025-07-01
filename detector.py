# === file: detector.py ===
"""Runtime detector used by FastAPI."""
import torch, pathlib
import torchvision.transforms as T
import numpy as np, cv2, base64
import torch.nn.functional as F
from config import WEIGHTS_DIR, IMG_SIZE, DEVICE
from model_utils import FeatureExtractor, get_backbone

class PaDiMDetector:
    def __init__(self, category: str):
        state = torch.load(WEIGHTS_DIR / f"{category}.pth", map_location=DEVICE)
        backbone, _ = get_backbone(category)
        backbone.to(DEVICE)
        self.extractor = FeatureExtractor(backbone, state["layers"])
        self.mean = state["mean"].to(DEVICE)
        self.cov_inv = state["cov_inv"].to(DEVICE)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])

    @torch.no_grad()
    def predict(self, img_bgr: np.ndarray):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inp = self.transform(img_rgb).unsqueeze(0).to(DEVICE)
        fmap = self.extractor(inp)           # [1,C,H,W]
        C = fmap.shape[1]
        fm = fmap.permute(0,2,3,1).reshape(-1, C)
        diff = fm - self.mean
        dist = (diff @ self.cov_inv * diff).sum(1)
        score = dist.max().item()
        return score