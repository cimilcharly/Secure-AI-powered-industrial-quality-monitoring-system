"""
Unpaired Dataset Loader for CycleGAN.
Loads patches from trainA (Nominal) and trainB (Damage) independently.
"""

import os
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class CycleGANDataset(Dataset):
    def __init__(self, root_dir, mode="train", transform=None):
        self.root_dir = Path(root_dir)
        self.mode = mode

        dir_a = self.root_dir / f"{mode}A"
        dir_b = self.root_dir / f"{mode}B"

        self.files_A = sorted(list(dir_a.glob("*.jpg")) + list(dir_a.glob("*.png")))
        self.files_B = sorted(list(dir_b.glob("*.jpg")) + list(dir_b.glob("*.png")))

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, index):
        file_a = self.files_A[index % len(self.files_A)]
        # Unpaired random sampling for domain B
        file_b = self.files_B[random.randint(0, len(self.files_B) - 1)]

        img_A = Image.open(file_a).convert("RGB")
        img_B = Image.open(file_b).convert("RGB")

        item_A = self.transform(img_A)
        item_B = self.transform(img_B)

        return {"A": item_A, "B": item_B, "path_A": str(file_a), "path_B": str(file_b)}
