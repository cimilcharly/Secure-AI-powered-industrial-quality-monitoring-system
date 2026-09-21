import cv2
import numpy as np
import os

ARTIFACT_DIR = r"C:\Users\HP\.gemini\antigravity-ide\brain\7c5aecdd-ac14-4e26-a4ad-68e0b6140d38"
STITCH_DIR = r"c:\Users\HP\Desktop\Dress Defect\data\ai_generated\stitch"

# Fine-tuned tight bounding boxes V3
CALIBRATED_BOXES_V3 = {
    # 1. Broken stitch (1264 x 848)
    # The broken stitch on beige back yoke is at (550, 160, 605, 200)
    'stitch_broken_back_yoke_seam.jpg': (550, 160, 605, 200),
    # Frayed hem on maroon bottom
    'stitch_broken_bottom_hem.jpg': (620, 715, 690, 765),
    # Frayed thread burst on black left side seam
    'stitch_broken_left_side_seam.jpg': (405, 490, 445, 575),
    # Dangling frayed threads on white sleeve
    'stitch_broken_left_sleeve_seam.jpg': (285, 495, 360, 565),
    # Frayed thread burst on navy right side seam
    'stitch_broken_right_side_seam.jpg': (830, 490, 870, 580),
    # Broken stitch on grey left shoulder seam
    'stitch_broken_shoulder_seam.jpg': (400, 150, 480, 210),

    # 2. Loose thread (1024 x 1024)
    # Wavy loose thread cluster on beige back yoke
    'stitch_loose_thread_back_yoke_seam.jpg': (590, 210, 680, 310),
    # Tangled loose threads emerging from grey bottom hem
    'stitch_loose_thread_bottom_hem.jpg': (460, 710, 610, 830),
    # Hanging thread from collar neckline
    'stitch_loose_thread_collar_seam.jpg': (445, 170, 515, 275),
    # Curled dangling thread from black cuff
    'stitch_loose_thread_cuff_seam.jpg': (770, 780, 875, 930),
    # Wavy loose thread curving along navy side seam
    'stitch_loose_thread_right_side_seam.jpg': (660, 540, 770, 730),
    # Loose thread cluster at blue tee armhole/shoulder seam
    'stitch_loose_thread_shoulder_seam.jpg': (240, 240, 340, 420),

    # 3. Open seam (1264 x 848)
    # Open dark gap on light blue back yoke
    'stitch_open_seam_back_yoke_seam_1789635312289.jpg': (525, 155, 680, 190),
    # Open separated seam on maroon collar
    'stitch_open_seam_collar_seam_1789635295024.jpg': (665, 125, 715, 210),
    # Dark open seam gap on white side seam
    'stitch_open_seam_left_side_seam_1789635243298.jpg': (425, 425, 460, 575),
    # Dark open underarm sleeve gap on navy shirt
    'stitch_open_seam_left_sleeve_seam_1789635273374.jpg': (875, 390, 960, 500),
    # Dark open seam gap on black side seam
    'stitch_open_seam_right_side_seam_1789635255619.jpg': (825, 430, 865, 585),
    # Dark open underarm sleeve gap on grey shirt
    'stitch_open_seam_right_sleeve_seam_1789635284867.jpg': (880, 430, 960, 540),

    # 4. Zigzag (1024 x 1024)
    # Irregular zigzag stitch pattern on light blue yoke
    'stitch_zigzag_back_yoke_seam.jpg': (315, 200, 450, 260),
    # Errant zigzag stitches on grey collar lapel edge
    'stitch_zigzag_collar_seam.jpg': (545, 130, 595, 215),
    # Errant zigzag stitches along black left side seam
    'stitch_zigzag_left_side_seam.jpg': (290, 520, 375, 670),
    # Errant zigzag stitch run along navy right side seam
    'stitch_zigzag_right_side_seam.jpg': (680, 470, 735, 610),
    # Massive zigzag stitch cluster on navy fabric
    'stitch_zigzag_shoulder_seam.jpg': (380, 220, 640, 640),
    # Errant zigzag stitches along white sleeve seam
    'stitch_zigzag_sleeve_seam.jpg': (110, 600, 195, 740),
}

crops_list = []
for fname, box in CALIBRATED_BOXES_V3.items():
    p = os.path.join(STITCH_DIR, fname)
    img = cv2.imread(p)
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))
    
    vis = img.copy()
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    pad = 50
    cx1 = max(0, x1 - pad)
    cy1 = max(0, y1 - pad)
    cx2 = min(w, x2 + pad)
    cy2 = min(h, y2 + pad)
    
    crop = vis[cy1:cy2, cx1:cx2].copy()
    crop_resized = cv2.resize(crop, (250, 250))
    cv2.putText(crop_resized, fname.replace('.jpg', '')[:22], (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    crops_list.append(crop_resized)

rows = []
for r in range(4):
    rows.append(np.hstack(crops_list[r*6 : (r+1)*6]))

final_vis = np.vstack(rows)
out_p = os.path.join(ARTIFACT_DIR, "all_24_calibrated_stitch_boxes_v3.jpg")
cv2.imwrite(out_p, final_vis)
print(f"Calibration preview v3 saved to {out_p}")
