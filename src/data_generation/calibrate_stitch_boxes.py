import cv2
import numpy as np
import os
import glob

ARTIFACT_DIR = r"C:\Users\HP\.gemini\antigravity-ide\brain\7c5aecdd-ac14-4e26-a4ad-68e0b6140d38"
STITCH_DIR = r"c:\Users\HP\Desktop\Dress Defect\data\ai_generated\stitch"

# Initial candidates based on visual inspection of grids
# Format: filename: (x1, y1, x2, y2)
CANDIDATES = {
    # 1. Broken stitch (shape: 1264x848)
    'stitch_broken_back_yoke_seam.jpg': (565, 80, 680, 115),      # let's verify exact position on 1264x848
    'stitch_broken_bottom_hem.jpg': (625, 420, 675, 455),
    'stitch_broken_left_side_seam.jpg': (970, 290, 995, 360),
    'stitch_broken_left_sleeve_seam.jpg': (270, 275, 340, 365),
    'stitch_broken_right_side_seam.jpg': (700, 300, 725, 370),
    'stitch_broken_shoulder_seam.jpg': (1000, 80, 1070, 125),
}

# Let's inspect where the defects actually are on the unresized images by computing difference or finding contours in the seam areas!
for name, p in [
    ('stitch_broken_back_yoke_seam.jpg', os.path.join(STITCH_DIR, 'stitch_broken_back_yoke_seam.jpg')),
    ('stitch_broken_bottom_hem.jpg', os.path.join(STITCH_DIR, 'stitch_broken_bottom_hem.jpg')),
    ('stitch_broken_left_side_seam.jpg', os.path.join(STITCH_DIR, 'stitch_broken_left_side_seam.jpg')),
    ('stitch_broken_left_sleeve_seam.jpg', os.path.join(STITCH_DIR, 'stitch_broken_left_sleeve_seam.jpg')),
    ('stitch_broken_right_side_seam.jpg', os.path.join(STITCH_DIR, 'stitch_broken_right_side_seam.jpg')),
    ('stitch_broken_shoulder_seam.jpg', os.path.join(STITCH_DIR, 'stitch_broken_shoulder_seam.jpg')),
]:
    img = cv2.imread(p)
    h, w = img.shape[:2]
    print(f"{name}: shape = {w}x{h}")
