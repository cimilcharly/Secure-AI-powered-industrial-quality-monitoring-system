"""
Standalone Dataset Builder for Dhakshina's 'Button' Class (Class 1 / Single-Class Detector).
Builds an independent, clean YOLOv8 dataset of exactly 200 images and YOLO labels,
properly split into Train (140), Val (40), and Test (20).

Subclass distribution across 3 button defect categories:
- missing: 67 images (47 train, 13 val, 7 test)
- loose: 67 images (47 train, 13 val, 7 test)
- wrong_color: 66 images (46 train, 14 val, 6 test)
Total: 200 images (140 train, 40 val, 20 test)
"""

import os
import json
import shutil
import random
import cv2
import numpy as np
from pathlib import Path

# Reproducibility
random.seed(42)
np.random.seed(42)

BASE_DIR = Path(r"c:\Users\HP\Desktop\Dress Defect")
BUTTON_MASTER_DIR = BASE_DIR / "data" / "ai_generated" / "button"
DATASET_DIR = BASE_DIR / "data" / "button_dataset"

def load_master_boxes():
    json_path = BUTTON_MASTER_DIR / "calibrated_master_boxes.json"
    assert json_path.exists(), f"Master boxes not found: {json_path}"
    with open(json_path, "r") as f:
        return json.load(f)

def transform_sample(image, box):
    """
    Applies affine, photometric, and optical transformations
    while analytically transforming the bounding box coordinates.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box

    # 1. Subtle Affine Transformation (rotation & translation)
    angle = random.uniform(-4.0, 4.0)
    scale = random.uniform(0.97, 1.03)
    dx = random.randint(-10, 10)
    dy = random.randint(-10, 10)

    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    M[0, 2] += dx
    M[1, 2] += dy

    aug_img = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # Transform 4 box corners with matrix M
    corners = np.array([
        [x1, y1, 1.0],
        [x2, y1, 1.0],
        [x2, y2, 1.0],
        [x1, y2, 1.0]
    ]).T

    transformed_corners = np.dot(M, corners)
    nx1 = np.min(transformed_corners[0, :])
    nx2 = np.max(transformed_corners[0, :])
    ny1 = np.min(transformed_corners[1, :])
    ny2 = np.max(transformed_corners[1, :])

    # 2. Random Horizontal Flip (50% probability)
    # Note: If flip is applied, shirt symmetry holds and button transfers horizontally
    flipped = random.random() > 0.5
    if flipped:
        aug_img = cv2.flip(aug_img, 1)
        fx1 = w - nx2
        fx2 = w - nx1
        nx1, nx2 = fx1, fx2

    # Clamp bounding box to image boundaries
    nx1 = max(0, min(w - 1, int(round(nx1))))
    ny1 = max(0, min(h - 1, int(round(ny1))))
    nx2 = max(0, min(w, int(round(nx2))))
    ny2 = max(0, min(h, int(round(ny2))))

    # 3. Photometric Adjustments (Brightness & Contrast)
    alpha = random.uniform(0.90, 1.12)
    beta = random.randint(-12, 12)
    aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)

    # 4. Subtle Color/Saturation Jitter in HSV
    if random.random() > 0.4:
        hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat_scale = random.uniform(0.93, 1.07)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
        aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 5. Gaussian Sensor Noise
    if random.random() > 0.3:
        noise = np.random.normal(0, random.uniform(2.0, 3.5), aug_img.shape).astype(np.float32)
        aug_img = np.clip(aug_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 6. Occasional subtle optical blur (20% probability)
    if random.random() > 0.8:
        aug_img = cv2.GaussianBlur(aug_img, (3, 3), 0.5)

    return aug_img, (nx1, ny1, nx2, ny2)

def to_yolo_format(box, width, height, class_id=0):
    """Converts pixel coordinates (x1, y1, x2, y2) to YOLO format string."""
    x1, y1, x2, y2 = box
    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    w = (x2 - x1) / float(width)
    h = (y2 - y1) / float(height)
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"

def build_dataset():
    print("=" * 65)
    print("BUILDING 200-IMAGE BUTTON DEFECT DATASET (DHAKSHINA'S CLASS)")
    print("=" * 65)

    master_boxes = load_master_boxes()

    # Clean existing directory
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    for split in ["train", "val", "test"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Subclass targets: missing=67, loose=67, wrong_color=66 -> exactly 200 images
    subclasses = {
        "missing": {"target": 67, "train": 47, "val": 13, "test": 7},
        "loose": {"target": 67, "train": 47, "val": 13, "test": 7},
        "wrong_color": {"target": 66, "train": 46, "val": 14, "test": 6}
    }

    stats = {
        "train": 0,
        "val": 0,
        "test": 0,
        "subclasses": {"missing": 0, "loose": 0, "wrong_color": 0}
    }

    for sub, config in subclasses.items():
        sub_dir = BUTTON_MASTER_DIR / sub
        seed_files = sorted(list(sub_dir.glob("*.png")))
        print(f"\nProcessing subclass: {sub.upper()} (Seeds: {len(seed_files)}, Target: {config['target']})")

        # Load seed images in memory
        seeds = []
        for sf in seed_files:
            img = cv2.imread(str(sf))
            box = master_boxes[sf.name]
            seeds.append((sf.name, img, box))

        # Generate target count for this subclass
        target_count = config["target"]
        train_count = config["train"]
        val_count = config["val"]
        test_count = config["test"]
        assert train_count + val_count + test_count == target_count

        sub_samples = []

        # 1. First include all original master seeds as baseline samples
        for idx, (fn, img, box) in enumerate(seeds):
            sub_samples.append((img.copy(), box, f"{sub}_master_{idx:03d}"))

        # 2. Augment by cycling seeds until target_count is reached
        seed_idx = 0
        while len(sub_samples) < target_count:
            fn, base_img, base_box = seeds[seed_idx % len(seeds)]
            aug_img, aug_box = transform_sample(base_img, base_box)
            aug_tag = f"{sub}_aug_{len(sub_samples):03d}"
            sub_samples.append((aug_img, aug_box, aug_tag))
            seed_idx += 1

        # Shuffle deterministically
        random.shuffle(sub_samples)

        # Partition into train / val / test
        train_slice = sub_samples[:train_count]
        val_slice = sub_samples[train_count:train_count + val_count]
        test_slice = sub_samples[train_count + val_count:]

        split_map = [
            ("train", train_slice),
            ("val", val_slice),
            ("test", test_slice)
        ]

        for split_name, slice_items in split_map:
            for img, box, tag in slice_items:
                h, w = img.shape[:2]
                out_name = f"button_{tag}.jpg"

                # Save image (high quality JPEG)
                img_out = DATASET_DIR / "images" / split_name / out_name
                cv2.imwrite(str(img_out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 96])

                # Save YOLO label
                lbl_out = DATASET_DIR / "labels" / split_name / out_name.replace(".jpg", ".txt")
                yolo_line = to_yolo_format(box, w, h, class_id=0)
                with open(lbl_out, "w") as lf:
                    lf.write(yolo_line)

                stats[split_name] += 1
                stats["subclasses"][sub] += 1

    # Write dataset.yaml
    dataset_yaml_content = f"""# YOLOv8 Button Defect Detection Dataset (Dhakshina's Class)
path: {DATASET_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: Button
"""
    yaml_path = DATASET_DIR / "dataset.yaml"
    with open(yaml_path, "w") as yf:
        yf.write(dataset_yaml_content)

    print("\n" + "=" * 65)
    print("DATASET BUILD COMPLETED SUCCESSFULLY!")
    print(f"Total Images: {stats['train'] + stats['val'] + stats['test']}")
    print(f"  - Train: {stats['train']} images (70%)")
    print(f"  - Val:   {stats['val']} images (20%)")
    print(f"  - Test:  {stats['test']} images (10%)")
    print(f"Subclass Breakdown: {stats['subclasses']}")
    print(f"Configuration: {yaml_path}")
    print("=" * 65)

if __name__ == "__main__":
    build_dataset()
