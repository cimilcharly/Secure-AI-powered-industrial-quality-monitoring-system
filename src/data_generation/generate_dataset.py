"""
High-Performance Synthetic Fabric Defect Dataset Generator & Auto-Labeler.
Generates 1,800 photorealistic 640x640 fabric images across:
  - 400 Damage (Class 0)
  - 400 Button (Class 1)
  - 400 Stitch (Class 2)
  - 400 Color (Class 3)
  - 200 Defect-Free Nominal (PatchCore)
Automatically generates YOLO normalized bounding-box .txt label files
and organizes into 70% Train, 15% Val, 15% Test partitions.
"""

import os
import random
import time
import cv2
import numpy as np

# Standard directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
ANNOTATED_DIR = os.path.join(DATA_DIR, "annotated")
RAW_NOMINAL_DIR = os.path.join(DATA_DIR, "raw", "defect_free")

SPLITS = ["train", "val", "test"]
for split in SPLITS:
    os.makedirs(os.path.join(ANNOTATED_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(ANNOTATED_DIR, "labels", split), exist_ok=True)
os.makedirs(RAW_NOMINAL_DIR, exist_ok=True)

# Curated fabric base colors (BGR format)
PALETTES = [
    (170, 160, 150),  # Light gray-linen
    (130, 140, 155),  # Slate denim
    (195, 185, 175),  # Khaki beige
    (120, 110, 100),  # Charcoal twill
    (150, 130, 120),  # Olive cotton
    (180, 170, 160),  # Cream wool
    (100, 105, 125),  # Deep navy
    (140, 125, 145),  # Heather heathered
    (160, 145, 135),  # Earth tan
    (185, 180, 175),  # Bleached cotton
]


def create_fabric_substrate(width=640, height=640, pattern_type="weave"):
    """Synthesizes realistic textile weave substrate with natural grain & lighting."""
    base_color = random.choice(PALETTES)
    img = np.full((height, width, 3), base_color, dtype=np.float32)

    # Micro-texture noise (fiber grain)
    grain = np.random.normal(0, 7.0, (height, width, 3)).astype(np.float32)
    img = np.clip(img + grain, 0, 255)

    # Weave structures
    if pattern_type == "twill":
        # Diagonal ribs (45 degree twill)
        y, x = np.ogrid[:height, :width]
        twill_pattern = (x + y) % 6
        twill_mask = (twill_pattern < 3).astype(np.float32) * 8.0 - 4.0
        img = np.clip(img + twill_mask[:, :, None], 0, 255)
    elif pattern_type == "knit":
        # Jersey knit loops
        y, x = np.ogrid[:height, :width]
        knit_mask = np.sin(x / 2.5) * np.cos(y / 3.0) * 6.0
        img = np.clip(img + knit_mask[:, :, None], 0, 255)
    else:
        # Plain grid weave
        img[::4, :, :] = np.clip(img[::4, :, :] - 5.0, 0, 255)
        img[:, ::4, :] = np.clip(img[:, ::4, :] + 5.0, 0, 255)

    # Gentle industrial lighting gradient
    center_x = random.randint(int(width * 0.3), int(width * 0.7))
    center_y = random.randint(int(height * 0.3), int(height * 0.7))
    y, x = np.ogrid[:height, :width]
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    vignette = 1.0 - (dist / (max(width, height) * 1.5)) * 0.12
    img = np.clip(img * vignette[:, :, None], 0, 255).astype(np.uint8)

    return img


def inject_damage(img):
    """Injects hole, tear, cut, or burn mark (Class 0). Returns (img, [x1, y1, x2, y2])."""
    h, w = img.shape[:2]
    cx = random.randint(int(w * 0.2), int(w * 0.8))
    cy = random.randint(int(h * 0.2), int(h * 0.8))
    dmg_type = random.choice(["tear", "hole", "cut", "burn"])

    if dmg_type == "hole":
        rx = random.randint(18, 45)
        ry = random.randint(15, 38)
        angle = random.randint(0, 180)
        # Scorched / frayed rim
        cv2.ellipse(img, (cx, cy), (rx + 6, ry + 6), angle, 0, 360, (30, 25, 20), -1)
        # Dark hollow center
        cv2.ellipse(img, (cx, cy), (rx, ry), angle, 0, 360, (12, 10, 10), -1)
        # Frayed fiber ends
        for _ in range(12):
            fx = cx + random.randint(-rx, rx)
            fy = cy + random.randint(-ry, ry)
            cv2.line(img, (fx, fy), (fx + random.randint(-6, 6), fy + random.randint(-6, 6)), (160, 150, 140), 1)
        x1, y1 = max(0, cx - rx - 8), max(0, cy - ry - 8)
        x2, y2 = min(w - 1, cx + rx + 8), min(h - 1, cy + ry + 8)

    elif dmg_type == "cut":
        length = random.randint(40, 90)
        angle_rad = random.uniform(0, np.pi)
        dx = int(length * np.cos(angle_rad) / 2)
        dy = int(length * np.sin(angle_rad) / 2)
        p1 = (cx - dx, cy - dy)
        p2 = (cx + dx, cy + dy)
        # Deep dark cut line
        cv2.line(img, p1, p2, (15, 12, 10), 3)
        # Frayed white border
        cv2.line(img, (p1[0] + 1, p1[1] + 1), (p2[0] + 1, p2[1] + 1), (200, 190, 180), 1)
        x1 = max(0, min(p1[0], p2[0]) - 6)
        y1 = max(0, min(p1[1], p2[1]) - 6)
        x2 = min(w - 1, max(p1[0], p2[0]) + 6)
        y2 = min(h - 1, max(p1[1], p2[1]) + 6)

    elif dmg_type == "burn":
        radius = random.randint(22, 50)
        # Brown heat ring
        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), radius + 15, (20, 35, 60), -1)
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
        # Charred center
        cv2.circle(img, (cx, cy), radius, (15, 18, 25), -1)
        x1, y1 = max(0, cx - radius - 18), max(0, cy - radius - 18)
        x2, y2 = min(w - 1, cx + radius + 18), min(h - 1, cy + radius + 18)

    else:  # tear
        pts = []
        radius = random.randint(25, 55)
        for i in range(8):
            ang = i * (2 * np.pi / 8) + random.uniform(-0.2, 0.2)
            r = radius * random.uniform(0.6, 1.2)
            pts.append([int(cx + r * np.cos(ang)), int(cy + r * np.sin(ang))])
        poly = np.array([pts], np.int32)
        cv2.fillPoly(img, poly, (15, 15, 15))
        bx, by, bw, bh = cv2.boundingRect(poly)
        x1, y1 = max(0, bx - 5), max(0, by - 5)
        x2, y2 = min(w - 1, bx + bw + 5), min(h - 1, by + bh + 5)

    return img, [x1, y1, x2, y2]


def inject_button(img):
    """Injects button anomaly: missing, cracked, wrong-color, or misaligned (Class 1)."""
    h, w = img.shape[:2]
    cx = random.randint(int(w * 0.25), int(w * 0.75))
    cy = random.randint(int(h * 0.25), int(h * 0.75))
    btn_type = random.choice(["missing", "cracked", "wrong_color", "misaligned"])

    # Placket background seam
    seam_val = int(max(0, int(img[0, cx, 0]) - 25))
    cv2.line(img, (cx, 0), (cx, h), (seam_val, seam_val, seam_val), 2)

    radius = random.randint(22, 36)

    if btn_type == "missing":
        # Loose stitches where button fell off
        cv2.circle(img, (cx, cy), 6, (40, 40, 40), -1)
        cv2.line(img, (cx - 8, cy - 8), (cx + 8, cy + 8), (220, 220, 220), 2)
        cv2.line(img, (cx - 8, cy + 8), (cx + 8, cy - 8), (220, 220, 220), 2)
        # Hanging loose thread
        pts = np.array([[cx, cy], [cx + 12, cy + 18], [cx + 8, cy + 32]], np.int32)
        cv2.polylines(img, [pts], False, (220, 220, 220), 2)
        x1, y1 = max(0, cx - radius), max(0, cy - radius)
        x2, y2 = min(w - 1, cx + radius), min(h - 1, cy + radius + 15)

    elif btn_type == "cracked":
        # Normal button with a fracture
        cv2.circle(img, (cx, cy), radius, (225, 225, 230), -1)
        cv2.circle(img, (cx, cy), radius, (140, 140, 145), 2)
        # 4 thread holes
        for ox, oy in [(-7, -7), (7, -7), (-7, 7), (7, 7)]:
            cv2.circle(img, (cx + ox, cy + oy), 3, (60, 60, 65), -1)
        # Jagged crack
        pts = np.array([[cx - radius + 3, cy - 5], [cx - 2, cy], [cx + 6, cy - 3], [cx + radius - 2, cy + 8]], np.int32)
        cv2.polylines(img, [pts], False, (25, 25, 30), 2)
        x1, y1 = max(0, cx - radius - 3), max(0, cy - radius - 3)
        x2, y2 = min(w - 1, cx + radius + 3), min(h - 1, cy + radius + 3)

    elif btn_type == "wrong_color":
        # Contrasting neon / dark mismatched button
        color = random.choice([(30, 40, 190), (180, 40, 40), (20, 160, 180), (30, 30, 35)])
        cv2.circle(img, (cx, cy), radius, color, -1)
        cv2.circle(img, (cx, cy), radius, (20, 20, 25), 2)
        for ox, oy in [(-6, -6), (6, -6), (-6, 6), (6, 6)]:
            cv2.circle(img, (cx + ox, cy + oy), 2, (240, 240, 240), -1)
        x1, y1 = max(0, cx - radius - 3), max(0, cy - radius - 3)
        x2, y2 = min(w - 1, cx + radius + 3), min(h - 1, cy + radius + 3)

    else:  # misaligned
        shift_x = cx + random.choice([-25, 25])
        cv2.circle(img, (shift_x, cy), radius, (215, 215, 220), -1)
        cv2.circle(img, (shift_x, cy), radius, (130, 130, 135), 2)
        x1, y1 = max(0, shift_x - radius - 3), max(0, cy - radius - 3)
        x2, y2 = min(w - 1, shift_x + radius + 3), min(h - 1, cy + radius + 3)

    return img, [x1, y1, x2, y2]


def inject_stitch(img):
    """Injects broken seam, open edge, loose loops, or skipped stitches (Class 2)."""
    h, w = img.shape[:2]
    stitch_type = random.choice(["broken", "open_seam", "loose_thread", "zigzag"])

    # Base seam path across image
    is_horiz = random.choice([True, False])
    if is_horiz:
        y_pos = random.randint(int(h * 0.3), int(h * 0.7))
        defect_start = random.randint(int(w * 0.25), int(w * 0.65))
        defect_len = random.randint(60, 130)

        # Baseline regular stitch lines
        for x in range(0, w, 14):
            if not (defect_start <= x <= defect_start + defect_len):
                cv2.line(img, (x, y_pos), (x + 8, y_pos), (230, 230, 100), 2)

        cx = defect_start + defect_len // 2
        cy = y_pos

        if stitch_type == "open_seam":
            # Dark open gap where two cloth pieces separated
            cv2.ellipse(img, (cx, cy), (defect_len // 2, 8), 0, 0, 360, (25, 20, 20), -1)
            x1, y1 = defect_start - 5, cy - 12
            x2, y2 = defect_start + defect_len + 5, cy + 12
        elif stitch_type == "loose_thread":
            # Wavy looped thread sticking out
            pts = np.array([
                [defect_start, cy],
                [defect_start + 20, cy - 35],
                [defect_start + 45, cy - 15],
                [defect_start + 70, cy - 40],
                [defect_start + defect_len, cy]
            ], np.int32)
            cv2.polylines(img, [pts], False, (240, 240, 80), 2)
            x1, y1 = defect_start - 5, cy - 45
            x2, y2 = defect_start + defect_len + 5, cy + 10
        else:  # broken or skipped
            cv2.line(img, (defect_start, cy - 4), (defect_start + defect_len, cy + 4), (50, 40, 30), 2)
            x1, y1 = defect_start - 5, cy - 8
            x2, y2 = defect_start + defect_len + 5, cy + 8

    else:
        x_pos = random.randint(int(w * 0.3), int(w * 0.7))
        defect_start = random.randint(int(h * 0.25), int(h * 0.65))
        defect_len = random.randint(60, 130)

        for y in range(0, h, 14):
            if not (defect_start <= y <= defect_start + defect_len):
                cv2.line(img, (x_pos, y), (x_pos, y + 8), (230, 230, 100), 2)

        cx = x_pos
        cy = defect_start + defect_len // 2

        if stitch_type == "open_seam":
            cv2.ellipse(img, (cx, cy), (8, defect_len // 2), 0, 0, 360, (25, 20, 20), -1)
            x1, y1 = cx - 12, defect_start - 5
            x2, y2 = cx + 12, defect_start + defect_len + 5
        else:
            pts = np.array([
                [cx, defect_start],
                [cx + 30, defect_start + 25],
                [cx - 15, defect_start + 55],
                [cx + 25, defect_start + 85],
                [cx, defect_start + defect_len]
            ], np.int32)
            cv2.polylines(img, [pts], False, (240, 240, 80), 2)
            x1, y1 = cx - 20, defect_start - 5
            x2, y2 = cx + 35, defect_start + defect_len + 5

    return img, [max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)]


def inject_color(img):
    """Injects dye spot, bleach patch, ink stain, or shade variation (Class 3)."""
    h, w = img.shape[:2]
    cx = random.randint(int(w * 0.25), int(w * 0.75))
    cy = random.randint(int(h * 0.25), int(h * 0.75))
    color_type = random.choice(["bleach", "dye_patch", "ink_stain", "shade_band"])

    radius = random.randint(35, 75)

    if color_type == "bleach":
        # Lightened circular patch with blurred fading boundary
        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), radius, (245, 245, 245), -1)
        img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
        x1, y1 = max(0, cx - radius), max(0, cy - radius)
        x2, y2 = min(w - 1, cx + radius), min(h - 1, cy + radius)

    elif color_type == "ink_stain":
        # Deep blue/black ink drip with satellite droplets
        ink_color = random.choice([(140, 30, 20), (20, 20, 140), (25, 25, 30)])
        cv2.circle(img, (cx, cy), radius // 2, ink_color, -1)
        for _ in range(5):
            ox = cx + random.randint(-radius, radius)
            oy = cy + random.randint(-radius, radius)
            cv2.circle(img, (ox, oy), random.randint(3, 8), ink_color, -1)
        x1, y1 = max(0, cx - radius - 5), max(0, cy - radius - 5)
        x2, y2 = min(w - 1, cx + radius + 5), min(h - 1, cy + radius + 5)

    elif color_type == "dye_patch":
        # Dark concentrated dye bleed
        overlay = img.copy()
        dye_color = (random.randint(20, 60), random.randint(20, 80), random.randint(140, 220))
        cv2.ellipse(overlay, (cx, cy), (radius, int(radius * 0.65)), random.randint(0, 180), 0, 360, dye_color, -1)
        img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
        x1, y1 = max(0, cx - radius - 5), max(0, cy - radius - 5)
        x2, y2 = min(w - 1, cx + radius + 5), min(h - 1, cy + radius + 5)

    else:  # shade_band
        overlay = img.copy()
        band_h = random.randint(60, 120)
        y1 = max(0, cy - band_h // 2)
        y2 = min(h - 1, cy + band_h // 2)
        cv2.rectangle(overlay, (0, y1), (w, y2), (random.randint(40, 80), random.randint(30, 70), random.randint(50, 90)), -1)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
        x1, x2 = int(w * 0.1), int(w * 0.9)

    return img, [x1, y1, x2, y2]


def save_sample(image, bbox, class_id, split, sample_idx, prefix):
    """Saves image and standard YOLO formatted .txt label."""
    img_name = f"{prefix}_{sample_idx:04d}.jpg"
    lbl_name = f"{prefix}_{sample_idx:04d}.txt"

    img_path = os.path.join(ANNOTATED_DIR, "images", split, img_name)
    lbl_path = os.path.join(ANNOTATED_DIR, "labels", split, lbl_name)

    cv2.imwrite(img_path, image)

    # Compute YOLO normalized bounding box format
    if bbox is not None and class_id is not None:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        bw = max(0.005, (x2 - x1) / w)
        bh = max(0.005, (y2 - y1) / h)
        bx = min(1.0, max(0.0, ((x1 + x2) / 2.0) / w))
        by = min(1.0, max(0.0, ((y1 + y2) / 2.0) / h))

        with open(lbl_path, "w") as f:
            f.write(f"{class_id} {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}\n")
    else:
        # Background / Defect-free image has empty label file
        with open(lbl_path, "w") as f:
            pass


def generate_full_dataset(target_per_category=400, nominal_count=200):
    """
    Main generator routine.
    Creates 400 * 4 = 1,600 defect samples + 200 nominal samples = 1,800 total.
    Applies 70% Train, 15% Val, 15% Test partition.
    """
    start_time = time.time()
    print("==================================================================")
    print("  GENERATING SYNTHETIC INDUSTRIAL FABRIC DEFECT DATASET (1,800 IMAGES)")
    print("==================================================================")

    categories = [
        ("damage", 0, inject_damage, target_per_category),
        ("button", 1, inject_button, target_per_category),
        ("stitch", 2, inject_stitch, target_per_category),
        ("color", 3, inject_color, target_per_category),
    ]

    total_generated = 0

    for cat_name, cls_id, inject_func, count in categories:
        print(f"Generating {count} samples for Class {cls_id} [{cat_name.upper()}]...")
        for i in range(1, count + 1):
            pattern = random.choice(["weave", "twill", "knit"])
            img = create_fabric_substrate(pattern_type=pattern)
            img, bbox = inject_func(img)

            # Determine split: 70% train, 15% val, 15% test
            r = i / count
            if r <= 0.70:
                split = "train"
            elif r <= 0.85:
                split = "val"
            else:
                split = "test"

            save_sample(img, bbox, cls_id, split, i, cat_name)
            total_generated += 1

    # Defect-free nominal fabrics
    print(f"Generating {nominal_count} Defect-Free Nominal fabrics for PatchCore...")
    for i in range(1, nominal_count + 1):
        pattern = random.choice(["weave", "twill", "knit"])
        img = create_fabric_substrate(pattern_type=pattern)

        # Save to raw nominal directory for PatchCore memory bank
        cv2.imwrite(os.path.join(RAW_NOMINAL_DIR, f"nominal_{i:04d}.jpg"), img)

        r = i / nominal_count
        if r <= 0.70:
            split = "train"
        elif r <= 0.85:
            split = "val"
        else:
            split = "test"

        save_sample(img, None, None, split, i, "nominal")
        total_generated += 1

    elapsed = round(time.time() - start_time, 2)
    print(f"\n[DONE] Successfully generated and auto-labeled {total_generated} images in {elapsed} seconds!")
    print(f"Dataset location: {ANNOTATED_DIR}")
    return total_generated


if __name__ == "__main__":
    generate_full_dataset()
