# === file: config.py ===
"""Global settings shared across modules."""
import pathlib, torch

ROOT = pathlib.Path("mvtec_anomaly_detection")  # dataset root
WEIGHTS_DIR = pathlib.Path("weights")           # where .pth files reside
IMG_SIZE = 256
BATCH = 32
SEED = 42

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

torch.manual_seed(SEED)