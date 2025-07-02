# === file: model_utils.py ===
"""Utilities: backbone loading, feature hooks, statistics."""
from typing import List
import torch, torchvision, torch.nn.functional as F
from tqdm import tqdm

ENHANCED_CATEGORIES = {"grid", "screw", "capsule", "pill", "cable", "toothbrush", "transistor"}

def get_backbone(category: str):
    if category in ENHANCED_CATEGORIES:
        model = torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V1")
        layers = ["layer1", "layer2", "layer3", "layer4"]
    else:
        model = torchvision.models.resnet34(weights="IMAGENET1K_V1")
        layers = ["layer1", "layer2", "layer3"]
    return model.eval(), layers

class FeatureExtractor:
    """Register forward hooks, return concatenated feature maps."""
    def __init__(self, backbone: torch.nn.Module, layer_names: List[str]):
        self.backbone = backbone
        self.layer_outputs = []
        self.hooks = [
            backbone._modules.get(n).register_forward_hook(lambda m, i, o: self.layer_outputs.append(o))
            for n in layer_names
        ]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.layer_outputs.clear()
        _ = self.backbone(x)
        maps = [f.detach() for f in self.layer_outputs]
        min_h = min(f.shape[2] for f in maps)
        maps = [F.interpolate(f, size=min_h, mode="bilinear", align_corners=False) for f in maps]
        cat = torch.cat(maps, 1)                # [B,C,H,W]
        return cat

    def close(self):
        for h in self.hooks:
            h.remove()