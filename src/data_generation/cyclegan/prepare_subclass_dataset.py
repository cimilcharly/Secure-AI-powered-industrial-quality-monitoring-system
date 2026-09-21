"""
Subclass-Aware Patch Extraction for Bharath's Damage Defect GAN.
Extracts 256x256 patches from ALL 24 Nominal shirts and ALL 33 Damaged shirts.
Organizes into 7 Damage Subclasses:
0: burn_mark
1: cut
2: frayed_edge
3: hole
4: rip
5: scratch
6: tear
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image

NOMINAL_DIR = Path("data/ai_generated/nominal")
DAMAGE_DIR = Path("data/ai_generated/damage")

OUTPUT_BASE = Path("data/cyclegan")
TRAIN_A = OUTPUT_BASE / "trainA"  # Nominal
TRAIN_B = OUTPUT_BASE / "trainB"  # Damage subclasses

SUBCLASSES = ["burn_mark", "cut", "frayed_edge", "hole", "rip", "scratch", "tear"]

PATCH_SIZE = 256
STRIDE = 128  # Dense stride for maximum coverage across all images

def get_subclass(filename):
    name = filename.lower()
    if "burn" in name:
        return "burn_mark"
    elif "cut" in name:
        return "cut"
    elif "fray" in name:
        return "frayed_edge"
    elif "hole" in name:
        return "hole"
    elif "rip" in name:
        return "rip"
    elif "scratch" in name:
        return "scratch"
    elif "tear" in name:
        return "tear"
    return "unknown"

def is_garment_patch(patch_np):
    """Filter out plain grey backdrop."""
    return np.std(patch_np) > 10.0

def extract_patches(img_path, patch_size=256, stride=128, max_patches=40):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            box = (x, y, x + patch_size, y + patch_size)
            patch = img.crop(box)
            if is_garment_patch(np.array(patch)):
                patches.append(patch)
                if len(patches) >= max_patches:
                    return patches
    return patches

def run_extraction():
    TRAIN_A.mkdir(parents=True, exist_ok=True)
    for sub in SUBCLASSES:
        (TRAIN_B / sub).mkdir(parents=True, exist_ok=True)

    nominal_files = sorted(list(NOMINAL_DIR.glob("*.jpg")) + list(NOMINAL_DIR.glob("*.png")))
    damage_files = sorted(list(DAMAGE_DIR.glob("*.jpg")) + list(DAMAGE_DIR.glob("*.png")))

    print(f"--- Domain A: Processing all {len(nominal_files)} Nominal shirts ---")
    count_a = 0
    for idx, f in enumerate(nominal_files):
        patches = extract_patches(f, PATCH_SIZE, STRIDE)
        for p_idx, p in enumerate(patches):
            p.save(TRAIN_A / f"nom_{idx:02d}_{p_idx:02d}.jpg", "JPEG", quality=95)
            count_a += 1
    print(f"Extracted {count_a} Domain A (Nominal) patches -> {TRAIN_A}")

    print(f"\n--- Domain B: Processing all {len(damage_files)} Damaged shirts ---")
    subclass_counts = {sub: 0 for sub in SUBCLASSES}
    for idx, f in enumerate(damage_files):
        sub = get_subclass(f.name)
        if sub not in SUBCLASSES:
            continue
        patches = extract_patches(f, PATCH_SIZE, STRIDE)
        for p_idx, p in enumerate(patches):
            target_dir = TRAIN_B / sub
            p.save(target_dir / f"{sub}_{idx:02d}_{p_idx:02d}.jpg", "JPEG", quality=95)
            subclass_counts[sub] += 1

    for sub, c in subclass_counts.items():
        print(f"  - Subclass '{sub}': {c} patches")

    total_b = sum(subclass_counts.values())
    print(f"Extracted {total_b} Domain B patches across 7 subclasses -> {TRAIN_B}")

if __name__ == "__main__":
    run_extraction()
