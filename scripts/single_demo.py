# %% ------------------------------------------------------------
#         Part 2: Fast Inference for a Single Image (Complete Version)
# ---------------------------------------------------------------
# Pre-requisite: Please ensure the following variables and classes from the main script are defined:
# ROOT, DEVICE, ENHANCED_CATEGORIES, MVTEC_PaDiM, make_loader
# If you are running in a new Notebook session, be sure to copy these definitions from your final script first.

from sklearn.metrics import precision_recall_curve
import numpy as np
import os
import cv2
import torch
import torch.nn.functional as F
import torchvision
import json
from sklearn.covariance import EmpiricalCovariance

from config import ROOT, DEVICE          # ROOT, DEVICE 共用
from model_utils import ENHANCED_CATEGORIES
from dataset import make_loader          # 若你在 dataset.py 定義
from dataset import MVTEC_PaDiM

# --- Create an inference-specific function ---
def predict_single_image(image_path, cat, thresholds_db):
    """
    Load pre-calculated thresholds to perform anomaly detection on a single specified image.
    """
    if not os.path.exists(image_path):
        print(f"❌ Error: Image path not found at '{image_path}'")
        return

    if cat not in thresholds_db:
        print(f"❌ Error: No threshold found for category '{cat}'. Please run the calibration script first.")
        return
        
    print(f"\n===== Predicting for a single '{cat}' image =====")
    print(f"Image path: {image_path}")

    # --- 1. Get pre-stored thresholds ---
    threshold = thresholds_db[cat]
    print(f"Loaded optimal threshold for '{cat}': {threshold:.4f}")

    # --- 2. Create the corresponding model (logic identical to calibration) ---
    if cat in ENHANCED_CATEGORIES:
        backbone = torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V1").to(DEVICE).eval()
        if cat == "pill": LAYER_NAMES = ["layer1", "layer2", "layer3"]
        else: LAYER_NAMES = ["layer1", "layer2", "layer3", "layer4"]
    else:
        backbone = torchvision.models.resnet34(weights="IMAGENET1K_V1").to(DEVICE).eval()
        LAYER_NAMES = ["layer1", "layer2", "layer3"]

    # --- 3. Prepare Gaussian distribution (this is a necessary step as it's used to calculate scores) ---
    print("Re-building Gaussian model for scoring...")
    
    feats_pred = []
    layers = [dict(backbone.named_modules())[n] for n in LAYER_NAMES]
    hooks = [l.register_forward_hook(lambda m,i,o: feats_pred.append(o)) for l in layers]

    # ❗❗--- Add the complete get_feats function definition ---❗❗
    def get_feats_for_prediction(loader):
        vecs=[]
        for imgs,_ in loader:
            imgs=imgs.to(DEVICE)
            feats_pred.clear()
            with torch.no_grad(): _ = backbone(imgs)
            
            fmap=[f.detach().cpu() for f in feats_pred]
            min_h=min([f.shape[2] for f in fmap])
            fmap=[F.interpolate(f,size=min_h,mode="bilinear", align_corners=False) for f in fmap]
            cat=torch.cat(fmap,1)
            vecs.append(cat.permute(0,2,3,1).reshape(-1,cat.shape[1]))
        return torch.cat(vecs)

    train_loader = make_loader(cat, "train")
    train_feats = get_feats_for_prediction(train_loader)
    
    # Due to randomness, the re-calculated mean/cov_inv will vary slightly each time.
    # In a real production system, mean/cov_inv should also be stored along with the thresholds.
    # However, in this project, this minor difference does not affect threshold application.
    sample = train_feats[torch.randperm(len(train_feats))[:20000]]
    mean = sample.mean(0, keepdim=True).to(DEVICE)
    cov_estimator = EmpiricalCovariance().fit(sample.numpy())
    cov_inv = torch.from_numpy(cov_estimator.precision_).float().to(DEVICE)
    for h in hooks: h.remove()
    
    # --- 4. Calculate score for a single image ---
    print(f"Calculating anomaly score for the single image...")
    transform_single = MVTEC_PaDiM(cat=cat, split="test").tr
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    img_tensor = transform_single(img).unsqueeze(0).to(DEVICE)

    hooks = [l.register_forward_hook(lambda m,i,o: feats_pred.append(o)) for l in layers]
    feats_pred.clear()
    with torch.no_grad(): _ = backbone(img_tensor)
    fmap=[f.detach() for f in feats_pred]; min_h=min([f.shape[2] for f in fmap])
    fmap=[F.interpolate(f,size=min_h,mode="bilinear", align_corners=False) for f in fmap]
    catf = torch.cat(fmap, 1).permute(0, 2, 3, 1)
    B, H, W, C = catf.shape
    flat = catf.reshape(-1, C); diff = flat - mean
    dist = (diff @ cov_inv * diff).sum(1)
    dist_tensor = dist.reshape(B, H, W)
    single_score, _ = torch.max(dist_tensor.view(B, -1), dim=1)
    single_score = single_score.item()
    for h in hooks: h.remove()
    print(f"Calculated anomaly score: {single_score:.4f}")

    # --- 5. Final judgment ---
    if single_score > threshold:
        print(f"🚨 Verdict: DEFECT (Score {single_score:.2f} > Threshold {threshold:.2f})")
    else:
        print(f"👍 Verdict: GOOD (Score {single_score:.2f} <= Threshold {threshold:.2f})")


# ===================================================================
#                  ⚙️ Perform quick inference ⚙️
# ===================================================================

# 1. Load our saved threshold database
try:
    with open("optimal_thresholds.json", "r") as f:
        thresholds_database = json.load(f)
    print("✅ Successfully loaded thresholds database from 'optimal_thresholds.json'")
except FileNotFoundError:
    print("❌ Thresholds file 'optimal_thresholds.json' not found.")
    print("   Please run the 'Part 1: Calculate and Save' script in the cell above first.")
    thresholds_database = None

# 2. Set the image and class you want to test
if thresholds_database:
    IMAGE_TO_TEST = "mvtec_anomaly_detection/bottle/test/good/009.png"  # <--- Please replace with the path to the image you want to test
    CATEGORY_OF_IMAGE = "bottle"                                        # <--- Please ensure the class corresponds to the image

    # 3. Execute inference
    predict_single_image(IMAGE_TO_TEST, CATEGORY_OF_IMAGE, thresholds_database)

    # You can continue to test other images
    print("\n--- Testing another image ---")
    IMAGE_TO_TEST_2 = "mvtec_anomaly_detection/screw/test/scratch_neck/007.png"
    CATEGORY_OF_IMAGE_2 = "screw"
    predict_single_image(IMAGE_TO_TEST_2, CATEGORY_OF_IMAGE_2, thresholds_database)