# === file: dataset.py ===  
"""MVTec AD dataset loader with Torch transforms."""
import cv2, torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from config import ROOT, IMG_SIZE

class MVTecPaDiM(Dataset):
    def __init__(self, category: str, split: str):
        assert split in {"train", "test"}
        self.samples = []
        base = ROOT / category / split
        subdirs = ["good"] if split == "train" else [d.name for d in base.iterdir() if d.is_dir()]
        for sd in subdirs:
            for p in (base / sd).glob("*.png"):
                label = 0 if sd == "good" else 1
                self.samples.append((p, label))

        self.tf = T.Compose([
            T.ToPILImage(),
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        return self.tf(img), label