# === file: model_utils.py ===  
"""
Utility functions:
    • FeatureExtractor  (unchanged – used in training / PaDiM)
    • load_model()      — memory-aware loader with auto-quantisation
    • infer_once()      — run inference once, then fully release RAM
"""

from __future__ import annotations

import os
import gc
import functools
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm

# ----------------------------------------------------------------------
#  Section 1 — original helper functions (kept for training scripts)
# ----------------------------------------------------------------------

ENHANCED_CATEGORIES = {
    "grid", "screw", "capsule", "pill",
    "cable", "toothbrush", "transistor",
}

def get_backbone(category: str):
    """Return a pretrained backbone and the layers to hook."""
    if category in ENHANCED_CATEGORIES:
        model = torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V1")
        layers = ["layer1", "layer2", "layer3", "layer4"]
    else:
        model = torchvision.models.resnet34(weights="IMAGENET1K_V1")
        layers = ["layer1", "layer2", "layer3"]
    return model.eval(), layers


class FeatureExtractor:
    """
    Register forward hooks and return concatenated feature maps.
    Used during PaDiM training.
    """
    def __init__(self, backbone: torch.nn.Module, layer_names: List[str]):
        self.backbone = backbone
        self.layer_outputs: List[torch.Tensor] = []
        self.hooks = [
            backbone._modules.get(n).register_forward_hook(
                lambda m, i, o: self.layer_outputs.append(o)
            )
            for n in layer_names
        ]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.layer_outputs.clear()
        _ = self.backbone(x)
        maps = [f.detach() for f in self.layer_outputs]
        min_h = min(f.shape[2] for f in maps)
        maps = [
            F.interpolate(
                f, size=min_h, mode="bilinear", align_corners=False
            )
            for f in maps
        ]
        return torch.cat(maps, 1)   # (B, C, H, W)

    def close(self) -> None:
        for h in self.hooks:
            h.remove()

# ----------------------------------------------------------------------
#  Section 2 — memory-aware model loading & inference
# ----------------------------------------------------------------------

WEIGHTS_DIR = Path("weights")      # adapt if your path differs
SIZE_LIMIT = 60 * 1024 * 1024      # >60 MB → treated as “heavy”

@functools.lru_cache(maxsize=1)    # keep only ONE model in RAM
def load_model(category: str) -> torch.nn.Module:
    """
    Lazy-load the model for a category.
    Heavy checkpoints are automatically converted to half-precision
    and dynamically quantised to int8 to save memory.
    """
    ckpt_path = WEIGHTS_DIR / f"{category}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")

    # Decide whether the checkpoint is “heavy”.
    is_heavy = (
        state.get("arch", "").lower().startswith("fastflow")
        or os.path.getsize(ckpt_path) > SIZE_LIMIT
    )

    if is_heavy:
        # Example: FastFlow – load backbone then quantise
        model = torchvision.models.wide_resnet50_2()
        model.load_state_dict(state["model"])
        model.half()  # fp16 first
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Conv2d, torch.nn.Linear},
            dtype=torch.qint8,
        )
    else:
        # Example: PaDiM – keep fp32
        model = torchvision.models.resnet34()
        model.load_state_dict(state["model"])

    model.eval()
    return model


def infer_once(
    category: str, img_tensor: torch.Tensor
) -> Tuple[float, float]:
    """
    Run inference once, obtain (score, threshold), then free memory.
    """
    model = load_model(category)

    with torch.inference_mode():
        out = model(img_tensor)         # adapt to your model’s forward
        # Assume `out` is a dict: {"score": s, "threshold": thr}
        # Modify here if your forward returns something else.
        score: float = float(out["score"])
        threshold: float = float(out["threshold"])

    # -------- full cleanup to avoid OOM on Render Starter --------
    del model
    torch.cuda.empty_cache()
    gc.collect()
    load_model.cache_clear()
    # -------------------------------------------------------------
    return score, threshold