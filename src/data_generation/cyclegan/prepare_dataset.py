"""
Prepare CycleGAN Dataset for Bharath's Damage Defect Generation.
Extracts 256x256 patches from:
- Domain A: 24 Nominal (Pristine) Shirts
- Domain B: 33 Damaged Shirts (Burns, Holes, Tears, Cuts, Scratches, Frays, Rips)
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image

NOMINAL_DIR = Path("data/ai_generated/nominal")
DAMAGE_DIR = Path("data/ai_generated/damage")

OUTPUT_BASE = Path("data/cyclegan")
TRAIN_A = OUTPUT_BASE / "trainA"
TRAIN_B = OUTPUT_BASE / "trainB"
TEST_A = OUTPUT_BASE / "testA"
TEST_B = OUTPUT_BASE / "testB"

PATCH_SIZE = 256
STRIDE = 160  # Overlapping stride for dense patch extraction

def is_garment_patch(patch_np):
    """
    Filter out pure solid background tiles (grey studio backdrop).
    A tile is kept if it has sufficient variance and color contrast.
    """
    # Background is typically grey (~ (200-240, 200-240, 200-240))
    std = np.std(patch_np)
    # If standard deviation is higher than threshold, it contains fabric texture/edges
    return std > 12.0

def extract_patches_from_image(img_path, patch_size=256, stride=160, max_patches=30):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            box = (x, y, x + patch_size, y + patch_size)
            patch = img.crop(box)
            patch_np = np.array(patch)
            if is_garment_patch(patch_np):
                patches.append(patch)
                if len(patches) >= max_patches:
                    return patches
    return patches

def run_preparation():
    for d in [TRAIN_A, TRAIN_B, TEST_A, TEST_B]:
        d.mkdir(parents=True, exist_ok=True)

    nominal_files = sorted(list(NOMINAL_DIR.glob("*.jpg")) + list(NOMINAL_DIR.glob("*.png")))
    damage_files = sorted(list(DAMAGE_DIR.glob("*.jpg")) + list(DAMAGE_DIR.glob("*.png")))

    print(f"Found {len(nominal_files)} Nominal images (Domain A)")
    print(f"Found {len(damage_files)} Damage images (Domain B)")

    # Extract Domain A Patches
    count_a = 0
    test_a_count = 0
    for idx, f in enumerate(nominal_files):
        # Save full downscaled 256x256 test sample for first 10
        if test_a_count < 10:
            with Image.open(f) as img:
                test_sample = img.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
                test_sample.save(TEST_A / f"testA_{test_a_count:02d}_{f.stem}.jpg", "JPEG", quality=95)
                test_a_count += 1

        patches = extract_patches_from_image(f, PATCH_SIZE, STRIDE, max_patches=25)
        for p_idx, patch in enumerate(patches):
            patch.save(TRAIN_A / f"patchA_{idx:02d}_{p_idx:02d}.jpg", "JPEG", quality=95)
            count_a += 1

    print(f"Extracted {count_a} Domain A (Nominal) training patches -> {TRAIN_A}")
    print(f"Saved {test_a_count} Domain A test samples -> {TEST_A}")

    # Extract Domain B Patches
    count_b = 0
    test_b_count = 0
    for idx, f in enumerate(damage_files):
        if test_b_count < 10:
            with Image.open(f) as img:
                test_sample = img.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
                test_sample.save(TEST_B / f"testB_{test_b_count:02d}_{f.stem}.jpg", "JPEG", quality=95)
                test_b_count += 1

        patches = extract_patches_from_image(f, PATCH_SIZE, STRIDE, max_patches=25)
        for p_idx, patch in enumerate(patches):
            patch.save(TRAIN_B / f"patchB_{idx:02d}_{p_idx:02d}.jpg", "JPEG", quality=95)
            count_b += 1

    print(f"Extracted {count_b} Domain B (Damage) training patches -> {TRAIN_B}")
    print(f"Saved {test_b_count} Domain B test samples -> {TEST_B}")

if __name__ == "__main__":
    run_preparation()
