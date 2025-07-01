# === file: train.py ===
"""Script to train PaDiM statistics and save weights."""
import json, torch, pathlib, numpy as np
from torch.utils.data import DataLoader
from sklearn.covariance import EmpiricalCovariance
from tqdm import tqdm
from config import ROOT, WEIGHTS_DIR, BATCH, DEVICE, SEED
from dataset import MVTecPaDiM
from model_utils import get_backbone, FeatureExtractor, ENHANCED_CATEGORIES

np.random.seed(SEED)

ALL_CATEGORIES = sorted([d.name for d in ROOT.iterdir() if d.is_dir()])
WEIGHTS_DIR.mkdir(exist_ok=True)

for cat in ALL_CATEGORIES:
    print(f"\n[Train] {cat}")
    backbone, layer_names = get_backbone(cat)
    backbone.to(DEVICE)
    extractor = FeatureExtractor(backbone, layer_names)

    train_loader = DataLoader(MVTecPaDiM(cat, "train"), batch_size=BATCH, shuffle=False)
    feats = []
    for imgs, _ in tqdm(train_loader, desc="extract"):
        imgs = imgs.to(DEVICE)
        with torch.no_grad():
            fmap = extractor(imgs)
        B, C, H, W = fmap.shape
        feats.append(fmap.permute(0, 2, 3, 1).reshape(-1, C).cpu())
    feats = torch.cat(feats)

    sample = feats[torch.randperm(len(feats))[:20000]]
    mean = sample.mean(0, keepdim=True)
    ec = EmpiricalCovariance().fit(sample.numpy())
    cov_inv = torch.from_numpy(ec.precision_).float()

    torch.save(
        {
            "arch": "wide_resnet50_2" if cat in ENHANCED_CATEGORIES else "resnet34",
            "layers": layer_names,
            "mean": mean,
            "cov_inv": cov_inv,
        },
        WEIGHTS_DIR / f"{cat}.pth",
    )
    extractor.close()
print("Done → all statistics saved in weights/")