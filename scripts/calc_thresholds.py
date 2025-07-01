# %% ------------------------------------------------------------
#         Part 1: Calculate and Save Optimal Thresholds for ALL Categories
# ---------------------------------------------------------------
# scripts/calc_thresholds.py 置頂
import pathlib, torch, torchvision, json, numpy as np
from tqdm import tqdm
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import precision_recall_curve

from config import ROOT, DEVICE          # ROOT, DEVICE 共用
from model_utils import ENHANCED_CATEGORIES
from dataset import make_loader          # 若你在 dataset.py 定義
                                         # (或 from train import make_loader)

# 自動取得所有類別
ALL_CATEGORIES = sorted([d.name for d in ROOT.iterdir() if d.is_dir()])

def calculate_all_thresholds():
    """
    Iterate through all MVTec classes, calculate the best F1-Score threshold for each class, and save to file.
    """
    # Ensure all necessary variables and functions are defined
    # (Inherited from your previous final integration script)
    
    all_thresholds = {}
    
    for cat in ALL_CATEGORIES:
        print(f"\n===== Calibrating Threshold for Category: [{cat}] =====")

        # --- 1. Load the best model corresponding to this class ---
        if cat in ENHANCED_CATEGORIES:
            backbone = torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V1").to(DEVICE).eval()
            if cat == "pill": LAYER_NAMES = ["layer1", "layer2", "layer3"]
            else: LAYER_NAMES = ["layer1", "layer2", "layer3", "layer4"]
        else:
            backbone = torchvision.models.resnet34(weights="IMAGENET1K_V1").to(DEVICE).eval()
            LAYER_NAMES = ["layer1", "layer2", "layer3"]
        
        # --- 2. Retrain the model to obtain Gaussian distribution ---
        feats_thresh = []
        layers = [dict(backbone.named_modules())[n] for n in LAYER_NAMES]
        hooks = [l.register_forward_hook(lambda m,i,o: feats_thresh.append(o)) for l in layers]

        def get_feats_thresh(loader):
            vecs=[]
            for imgs,_ in loader:
                imgs=imgs.to(DEVICE); feats_thresh.clear()
                with torch.no_grad(): _ = backbone(imgs)
                fmap=[f.detach().cpu() for f in feats_thresh]; min_h=min([f.shape[2] for f in fmap]); fmap=[F.interpolate(f,size=min_h,mode="bilinear", align_corners=False) for f in fmap]
                vecs.append(torch.cat(fmap,1).permute(0,2,3,1).reshape(-1,torch.cat(fmap,1).shape[1]))
            return torch.cat(vecs)

        train_loader = make_loader(cat, "train")
        train_feats = get_feats_thresh(train_loader)
        sample = train_feats[torch.randperm(len(train_feats))[:20000]]
        mean = sample.mean(0, keepdim=True).to(DEVICE)
        cov_estimator = EmpiricalCovariance().fit(sample.numpy())
        cov_inv = torch.from_numpy(cov_estimator.precision_).float().to(DEVICE)
        for h in hooks: h.remove()

        # --- 3. Obtain scores and labels on the entire test set ---
        hooks = [l.register_forward_hook(lambda m,i,o: feats_thresh.append(o)) for l in layers]
        test_loader = make_loader(cat, "test")
        all_scores, all_labels = [], []
        for imgs, lbls in tqdm(test_loader, desc=f"Scoring {cat}"):
            imgs=imgs.to(DEVICE); feats_thresh.clear()
            with torch.no_grad(): _ = backbone(imgs)
            fmap=[f.detach() for f in feats_thresh]; min_h=min([f.shape[2] for f in fmap])
            fmap=[F.interpolate(f,size=min_h,mode="bilinear", align_corners=False) for f in fmap]
            catf = torch.cat(fmap, 1).permute(0, 2, 3, 1)
            B, H, W, C = catf.shape
            flat = catf.reshape(-1, C); diff = flat - mean
            dist = (diff @ cov_inv * diff).sum(1)
            dist_tensor = dist.reshape(B, H, W)
            dist_img, _ = torch.max(dist_tensor.view(B, -1), dim=1)
            all_scores.extend(dist_img.cpu().numpy())
            all_labels.extend(lbls.numpy())
        for h in hooks: h.remove()
        
        # --- 4. Calculate and save the optimal threshold ---
        precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_threshold = thresholds[np.argmax(f1_scores)]
        all_thresholds[cat] = float(best_threshold) # Convert to float for easy JSON storage
        print(f"✅ Optimal threshold for '{cat}': {best_threshold:.4f}")

    # --- 5. Write all thresholds to file ---
    with open("optimal_thresholds.json", "w") as f:
        json.dump(all_thresholds, f, indent=2, sort_keys=True)
    print("\n\n✅ All thresholds have been calculated and saved to 'optimal_thresholds.json'")
    
    return all_thresholds

# --- Perform calibration ---
all_thresholds = calculate_all_thresholds()