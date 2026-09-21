"""
Standalone Dataset Builder for Cimil's 'Stitch' Class (Class 2 / Single-Class Detector).
Builds an independent, clean YOLOv8 dataset of exactly 200 images and YOLO labels,
properly split into Train (140), Val (40), and Test (20).

Contains calibrated ground-truth bounding boxes across all 4 stitch subclasses:
- broken (50 images: 35 train, 10 val, 5 test)
- loose_thread (50 images: 35 train, 10 val, 5 test)
- open_seam (50 images: 35 train, 10 val, 5 test)
- zigzag (50 images: 35 train, 10 val, 5 test)
Total: 200 images (140 train, 40 val, 20 test)
"""

import os
import shutil
import random
import cv2
import numpy as np

# Seed for reproducible split and transformations
random.seed(42)
np.random.seed(42)

# Paths
BASE_DIR = r"c:\Users\HP\Desktop\Dress Defect"
STITCH_MASTER_DIR = os.path.join(BASE_DIR, "data", "ai_generated", "stitch")
DATASET_DIR = os.path.join(BASE_DIR, "data", "stitch_dataset")

# Calibrated ground-truth bounding boxes for all 24 master seed images (x1, y1, x2, y2)
MASTER_BOXES = {
    # 1. Broken stitch (1264 x 848)
    'stitch_broken_back_yoke_seam.jpg': (550, 160, 605, 200),
    'stitch_broken_bottom_hem.jpg': (620, 715, 690, 765),
    'stitch_broken_left_side_seam.jpg': (405, 490, 445, 575),
    'stitch_broken_left_sleeve_seam.jpg': (285, 495, 360, 565),
    'stitch_broken_right_side_seam.jpg': (830, 490, 870, 580),
    'stitch_broken_shoulder_seam.jpg': (400, 150, 480, 210),

    # 2. Loose thread (1024 x 1024)
    'stitch_loose_thread_back_yoke_seam.jpg': (590, 210, 680, 310),
    'stitch_loose_thread_bottom_hem.jpg': (460, 710, 610, 830),
    'stitch_loose_thread_collar_seam.jpg': (445, 170, 515, 275),
    'stitch_loose_thread_cuff_seam.jpg': (770, 780, 875, 930),
    'stitch_loose_thread_right_side_seam.jpg': (660, 540, 770, 730),
    'stitch_loose_thread_shoulder_seam.jpg': (240, 240, 340, 420),

    # 3. Open seam (1264 x 848)
    'stitch_open_seam_back_yoke_seam_1789635312289.jpg': (525, 155, 680, 190),
    'stitch_open_seam_collar_seam_1789635295024.jpg': (665, 125, 715, 210),
    'stitch_open_seam_left_side_seam_1789635243298.jpg': (425, 425, 460, 575),
    'stitch_open_seam_left_sleeve_seam_1789635273374.jpg': (875, 390, 960, 500),
    'stitch_open_seam_right_side_seam_1789635255619.jpg': (825, 430, 865, 585),
    'stitch_open_seam_right_sleeve_seam_1789635284867.jpg': (880, 430, 960, 540),

    # 4. Zigzag (1024 x 1024)
    'stitch_zigzag_back_yoke_seam.jpg': (315, 200, 450, 260),
    'stitch_zigzag_collar_seam.jpg': (545, 130, 595, 215),
    'stitch_zigzag_left_side_seam.jpg': (290, 520, 375, 670),
    'stitch_zigzag_right_side_seam.jpg': (675, 470, 740, 610),
    'stitch_zigzag_shoulder_seam.jpg': (380, 220, 640, 640),
    'stitch_zigzag_sleeve_seam.jpg': (110, 600, 195, 740),
}

SUBCLASSES = ["broken", "loose_thread", "open_seam", "zigzag"]

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
    dx = random.randint(-12, 12)
    dy = random.randint(-12, 12)

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
    flipped = random.random() > 0.5
    if flipped:
        aug_img = cv2.flip(aug_img, 1)
        fx1 = w - nx2
        fx2 = w - nx1
        nx1, nx2 = fx1, fx2

    # Clamp bounding box to image boundaries
    nx1 = max(0, min(w - 1, int(round(nx1))))
    ny1 = max(0, min(h - 1, int(round(ny1))))
    nx2 = max(nx1 + 2, min(w, int(round(nx2))))
    ny2 = max(ny1 + 2, min(h, int(round(ny2))))

    # 3. Photometric Adjustments (Brightness & Contrast)
    alpha = random.uniform(0.88, 1.14)
    beta = random.randint(-14, 14)
    aug_img = cv2.convertScaleAbs(aug_img, alpha=alpha, beta=beta)

    # 4. Subtle Color/Saturation Jitter in HSV
    if random.random() > 0.4:
        hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.float32)
        sat_scale = random.uniform(0.92, 1.08)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
        aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 5. Gaussian Sensor Noise (Proper float32 clipping to prevent uint8 underflow)
    if random.random() > 0.3:
        noise = np.random.normal(0, random.uniform(2.0, 4.0), aug_img.shape).astype(np.float32)
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
    print("BUILDING 200-IMAGE STITCH DATASET FOR CIMIL'S CLASS (CLASS 2)")
    print("=" * 65)

    # Create clean directory structure
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)

    # Exactly 50 images per subclass x 4 subclasses = 200 images
    # Split per subclass: 35 Train (70%), 10 Val (20%), 5 Test (10%)
    # Total: 140 Train, 40 Val, 20 Test
    train_set = []
    val_set = []
    test_set = []

    for subclass in SUBCLASSES:
        subclass_masters = [k for k in MASTER_BOXES.keys() if f"stitch_{subclass}_" in k]
        subclass_masters.sort()
        print(f"\nProcessing subclass '{subclass}': {len(subclass_masters)} master seeds found")

        # 1. Add all master pristine seeds
        sub_samples = []
        for master_file in subclass_masters:
            master_path = os.path.join(STITCH_MASTER_DIR, master_file)
            img = cv2.imread(master_path)
            box = MASTER_BOXES[master_file]
            clean_name = master_file.replace(".jpg", "").replace("stitch_", "").replace("1789635", "")
            sub_samples.append({
                'image': img,
                'box': box,
                'subclass': subclass,
                'name': f"stitch_{subclass}_{clean_name}_master"
            })

        # 2. Add augmented variations to reach exactly 50 images
        aug_idx = 1
        while len(sub_samples) < 50:
            master_file = subclass_masters[(aug_idx - 1) % len(subclass_masters)]
            master_path = os.path.join(STITCH_MASTER_DIR, master_file)
            base_img = cv2.imread(master_path)
            box = MASTER_BOXES[master_file]
            clean_name = master_file.replace(".jpg", "").replace("stitch_", "").replace("1789635", "")

            aug_img, aug_box = transform_sample(base_img, box)
            sub_samples.append({
                'image': aug_img,
                'box': aug_box,
                'subclass': subclass,
                'name': f"stitch_{subclass}_{clean_name}_aug_{aug_idx:02d}"
            })
            aug_idx += 1

        # Shuffle subclass samples
        random.shuffle(sub_samples)

        # Stratified Split: 35 train, 10 val, 5 test
        sub_train = sub_samples[:35]
        sub_val = sub_samples[35:45]
        sub_test = sub_samples[45:50]

        train_set.extend(sub_train)
        val_set.extend(sub_val)
        test_set.extend(sub_test)

        print(f"  -> Generated 50 samples: 35 Train, 10 Val, 5 Test")

    # Shuffle sets
    random.shuffle(train_set)
    random.shuffle(val_set)
    random.shuffle(test_set)

    print("\n" + "=" * 65)
    print(f"FINAL DATASET SPLIT: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    print(f"TOTAL IMAGES: {len(train_set) + len(val_set) + len(test_set)}")
    print("=" * 65)

    # Save to disk
    splits = [("train", train_set), ("val", val_set), ("test", test_set)]
    for split_name, samples in splits:
        img_dir = os.path.join(DATASET_DIR, "images", split_name)
        lbl_dir = os.path.join(DATASET_DIR, "labels", split_name)

        for s in samples:
            img = s['image']
            box = s['box']
            h, w = img.shape[:2]
            fname = s['name']

            img_path = os.path.join(img_dir, f"{fname}.jpg")
            lbl_path = os.path.join(lbl_dir, f"{fname}.txt")

            cv2.imwrite(img_path, img)
            yolo_line = to_yolo_format(box, w, h, class_id=0)
            with open(lbl_path, "w") as f:
                f.write(yolo_line)

    # Write dataset.yaml
    yaml_content = f"""# Stitch Defect Detection Dataset (Cimil's Class)
path: {DATASET_DIR.replace(chr(92), '/')}
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: Stitch
"""
    yaml_path = os.path.join(DATASET_DIR, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n[OK] dataset.yaml written successfully at: {yaml_path}")
    print(f"[OK] 200 images successfully generated in: {DATASET_DIR}")

if __name__ == "__main__":
    build_dataset()
