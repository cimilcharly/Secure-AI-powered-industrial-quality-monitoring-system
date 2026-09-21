import cv2
import numpy as np
import os

ARTIFACT_DIR = r"C:\Users\HP\.gemini\antigravity-ide\brain\7c5aecdd-ac14-4e26-a4ad-68e0b6140d38"
STITCH_DIR = r"c:\Users\HP\Desktop\Dress Defect\data\ai_generated\stitch"

# Exact calibrated bounding boxes for all 24 master seed images: (x1, y1, x2, y2)
CALIBRATED_BOXES = {
    # 1. Broken stitch (1264 x 848)
    'stitch_broken_back_yoke_seam.jpg': (285, 155, 375, 195),
    'stitch_broken_bottom_hem.jpg': (695, 725, 745, 765),
    'stitch_broken_left_side_seam.jpg': (430, 495, 460, 580),
    'stitch_broken_left_sleeve_seam.jpg': (265, 500, 350, 575),
    'stitch_broken_right_side_seam.jpg': (825, 495, 855, 580),
    'stitch_broken_shoulder_seam.jpg': (425, 185, 495, 240),

    # 2. Loose thread (1024 x 1024)
    'stitch_loose_thread_back_yoke_seam.jpg': (640, 165, 730, 260),
    'stitch_loose_thread_bottom_hem.jpg': (465, 700, 605, 805),
    'stitch_loose_thread_collar_seam.jpg': (445, 170, 515, 275),
    'stitch_loose_thread_cuff_seam.jpg': (875, 810, 960, 925),
    'stitch_loose_thread_right_side_seam.jpg': (675, 530, 775, 735),
    'stitch_loose_thread_shoulder_seam.jpg': (275, 240, 360, 415),

    # 3. Open seam (1264 x 848)
    'stitch_open_seam_back_yoke_seam_1789635312289.jpg': (525, 155, 680, 190),
    'stitch_open_seam_collar_seam_1789635295024.jpg': (665, 125, 715, 210),
    'stitch_open_seam_left_side_seam_1789635243298.jpg': (425, 425, 460, 575),
    'stitch_open_seam_left_sleeve_seam_1789635273374.jpg': (840, 430, 920, 535),
    'stitch_open_seam_right_side_seam_1789635255619.jpg': (825, 430, 865, 585),
    'stitch_open_seam_right_sleeve_seam_1789635284867.jpg': (840, 430, 930, 545),

    # 4. Zigzag (1024 x 1024)
    'stitch_zigzag_back_yoke_seam.jpg': (315, 140, 450, 195),
    'stitch_zigzag_collar_seam.jpg': (520, 120, 575, 185),
    'stitch_zigzag_left_side_seam.jpg': (340, 510, 395, 660),
    'stitch_zigzag_right_side_seam.jpg': (645, 465, 685, 600),
    'stitch_zigzag_shoulder_seam.jpg': (380, 220, 640, 640),
    'stitch_zigzag_sleeve_seam.jpg': (210, 600, 295, 740),
}

# Verify and generate visual confirmation for all 24
crops_list = []
for fname, box in CALIBRATED_BOXES.items():
    p = os.path.join(STITCH_DIR, fname)
    img = cv2.imread(p)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    
    # Check bounds
    assert 0 <= x1 < x2 <= w, f"Invalid x in {fname}: {x1}, {x2}, w={w}"
    assert 0 <= y1 < y2 <= h, f"Invalid y in {fname}: {y1}, {y2}, h={h}"
    
    # Draw green box on copy
    vis = img.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Zoomed crop with context (100px padding around box)
    pad = 60
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(w, x2 + pad)
    cy2 = min(h, y2 + pad)
    
    crop = vis[cy1:cy2, cx1:cx2].copy()
    crop_resized = cv2.resize(crop, (250, 250))
    cv2.putText(crop_resized, fname.replace('.jpg', '')[:22], (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    crops_list.append(crop_resized)

# Assemble into 4 rows of 6
rows = []
for r in range(4):
    rows.append(np.hstack(crops_list[r*6 : (r+1)*6]))

final_vis = np.vstack(rows)
out_p = os.path.join(ARTIFACT_DIR, "all_24_calibrated_stitch_boxes.jpg")
cv2.imwrite(out_p, final_vis)
print(f"Calibration preview saved to {out_p}")
