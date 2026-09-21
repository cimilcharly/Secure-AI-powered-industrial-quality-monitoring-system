"""
Industrial Augmentation & Synthetic Expansion for Bharath's 'Damage' Subclasses.
Generates 10 photorealistic augmented images and YOLO bounding box labels
for each of the 7 damage subclasses (70 images total).
Subclasses: burn_mark, cut, frayed_edge, hole, rip, scratch, tear
"""

import os
import random
import cv2
import numpy as np

# Seed for reproducibility while preserving diversity
random.seed(42)
np.random.seed(42)

# Paths
DAMAGE_DIR = r"c:\Users\HP\Desktop\Dress Defect\data\ai_generated\damage"
OUTPUT_DIR = os.path.join(DAMAGE_DIR, "augmented")
PREVIEW_DIR = r"C:\Users\HP\.gemini\antigravity-ide\brain\7c5aecdd-ac14-4e26-a4ad-68e0b6140d38\damage_verification"

# Calibrated ground-truth bounding boxes for all 28 master seed images (x1, y1, x2, y2)
MASTER_BOXES = {
    # Burn mark
    'damage_burn_mark_black_lower_left_front_panel.jpg': (380, 620, 485, 730),
    'damage_burn_mark_grey_right_sleeve_forearm.jpg': (765, 500, 855, 615),
    'damage_burn_mark_navy_upper_right_back_panel.jpg': (560, 245, 640, 335),
    'damage_burn_mark_white_center_chest_area.jpg': (525, 335, 625, 465),

    # Cut
    'damage_cut_black_lower_left_front_panel.jpg': (400, 630, 440, 720),
    'damage_cut_grey_right_sleeve_forearm.jpg': (740, 510, 875, 560),
    'damage_cut_navy_upper_right_back_panel.jpg': (555, 320, 665, 355),
    'damage_cut_white_center_chest_area.jpg': (455, 410, 605, 445),

    # Frayed edge
    'damage_frayed_edge_black_lower_left_front_panel.jpg': (420, 635, 495, 730),
    'damage_frayed_edge_grey_right_sleeve_forearm.jpg': (695, 505, 875, 680),
    'damage_frayed_edge_navy_upper_right_back_panel.jpg': (575, 270, 685, 440),
    'damage_frayed_edge_white_center_chest_area.jpg': (510, 390, 630, 520),

    # Hole
    'damage_hole_black_lower_left_front_panel.jpg': (365, 700, 415, 760),
    'damage_hole_grey_right_sleeve_forearm.jpg': (720, 500, 770, 560),
    'damage_hole_navy_upper_right_back_panel.jpg': (610, 330, 670, 400),
    'damage_hole_white_center_chest_area.jpg': (595, 420, 660, 490),

    # Rip
    'damage_rip_black_lower_left_front_panel.jpg': (270, 740, 410, 950),
    'damage_rip_grey_right_sleeve_forearm.jpg': (900, 390, 1030, 510),
    'damage_rip_navy_upper_right_back_panel.jpg': (740, 200, 820, 420),
    'damage_rip_white_center_chest_area.jpg': (525, 485, 620, 585),

    # Scratch
    'damage_scratch_black_lower_left_front_panel.jpg': (330, 680, 440, 835),
    'damage_scratch_grey_right_sleeve_forearm.jpg': (740, 460, 850, 605),
    'damage_scratch_navy_upper_right_back_panel.jpg': (575, 265, 650, 390),
    'damage_scratch_white_center_chest_area.jpg': (440, 425, 630, 560),

    # Tear
    'damage_tear_black_lower_left_front_panel.jpg': (390, 635, 470, 805),
    'damage_tear_grey_right_sleeve_forearm.jpg': (750, 525, 830, 640),
    'damage_tear_navy_upper_right_back_panel.jpg': (580, 275, 660, 440),
    'damage_tear_white_center_chest_area.jpg': (545, 455, 720, 605),
}

SUBCLASSES = ["burn_mark", "cut", "frayed_edge", "hole", "rip", "scratch", "tear"]

def transform_sample(image, box):
    """
    Applies affine, photometric, and optical transformations
    while analytically transforming the bounding box coordinates.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box

    # 1. Subtle Affine Transformation (rotation & translation)
    angle = random.uniform(-4.5, 4.5)
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
    ]).T  # shape (3, 4)

    transformed_corners = np.dot(M, corners)  # shape (2, 4)
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
    nx2 = max(0, min(w, int(round(nx2))))
    ny2 = max(0, min(h, int(round(ny2))))

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
    """Converts pixel coordinates (x1, y1, x2, y2) to YOLO format."""
    x1, y1, x2, y2 = box
    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    w = (x2 - x1) / float(width)
    h = (y2 - y1) / float(height)
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)

    print("=" * 60)
    print("Starting Industrial Augmentation for Damage Subclasses...")
    print("=" * 60)

    total_images_generated = 0
    verification_samples = []

    for subclass in SUBCLASSES:
        # Find the 4 master images for this subclass
        subclass_masters = [k for k in MASTER_BOXES.keys() if f"damage_{subclass}_" in k]
        subclass_masters.sort()

        if len(subclass_masters) != 4:
            print(f"Warning: Expected 4 masters for {subclass}, found {len(subclass_masters)}")

        # Plan 10 variations:
        # master[0] (black): 3 variations
        # master[1] (grey): 3 variations
        # master[2] (navy): 2 variations
        # master[3] (white): 2 variations
        variation_plan = [
            (subclass_masters[0], 3),
            (subclass_masters[1], 3),
            (subclass_masters[2], 2),
            (subclass_masters[3], 2),
        ]

        subclass_out_dir = os.path.join(OUTPUT_DIR, subclass)
        os.makedirs(subclass_out_dir, exist_ok=True)

        sample_idx = 1
        print(f"\nGenerating 10 images for subclass '{subclass}'...")

        for master_file, count in variation_plan:
            master_path = os.path.join(DAMAGE_DIR, master_file)
            base_img = cv2.imread(master_path)
            if base_img is None:
                print(f"Error loading {master_path}")
                continue

            box = MASTER_BOXES[master_file]
            color = [c for c in ['black', 'grey', 'navy', 'white'] if c in master_file][0]

            for _ in range(count):
                aug_img, (nx1, ny1, nx2, ny2) = transform_sample(base_img, box)
                h, w = aug_img.shape[:2]

                base_name = f"damage_{subclass}_{color}_aug_{sample_idx:02d}"
                img_path = os.path.join(subclass_out_dir, f"{base_name}.jpg")
                txt_path = os.path.join(subclass_out_dir, f"{base_name}.txt")

                # Save image and label
                cv2.imwrite(img_path, aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                yolo_line = to_yolo_format((nx1, ny1, nx2, ny2), w, h, class_id=0)
                with open(txt_path, "w") as f:
                    f.write(yolo_line)

                # Save first sample of each subclass for visual verification
                if sample_idx == 1:
                    preview_img = aug_img.copy()
                    cv2.rectangle(preview_img, (nx1, ny1), (nx2, ny2), (0, 0, 255), 3)
                    label_text = f"{subclass.upper()} (Class 0)"
                    cv2.putText(preview_img, label_text, (nx1, max(30, ny1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    preview_path = os.path.join(PREVIEW_DIR, f"verify_{subclass}.jpg")
                    cv2.imwrite(preview_path, preview_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    verification_samples.append((subclass, preview_path))

                total_images_generated += 1
                sample_idx += 1

        print(f"  [OK] Successfully created 10 samples for '{subclass}' in {subclass_out_dir}")

    # Generate full 7-subclass verification collage
    print("\nGenerating full 7-subclass verification collage...")
    collage_imgs = []
    for subclass, preview_path in verification_samples:
        p_img = cv2.imread(preview_path)
        p_thumb = cv2.resize(p_img, (384, 384))
        collage_imgs.append(p_thumb)

    row1 = np.hstack(collage_imgs[:4])
    row2 = np.hstack(collage_imgs[4:] + [np.zeros_like(collage_imgs[0])])
    full_verification_grid = np.vstack([row1, row2])
    collage_path = os.path.join(PREVIEW_DIR, "all_7_subclasses_verification.jpg")
    cv2.imwrite(collage_path, full_verification_grid, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print("=" * 60)
    print(f"COMPLETE! Generated {total_images_generated} augmented images & labels.")
    print(f"Verification collage saved at: {collage_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
