import cv2
import numpy as np
import os

ARTIFACT_DIR = r"C:\Users\HP\.gemini\antigravity-ide\brain\7c5aecdd-ac14-4e26-a4ad-68e0b6140d38"
STITCH_DIR = r"c:\Users\HP\Desktop\Dress Defect\data\ai_generated\stitch"

# For each image, let's specify an approximate center (cx, cy) to crop a 250x250 window
# with coordinate axes drawn so we can see the exact bounding box down to pixel precision!
CENTERS = {
    # Broken (1264x848)
    'stitch_broken_back_yoke_seam.jpg': (580, 105),
    'stitch_broken_bottom_hem.jpg': (650, 435),
    'stitch_broken_left_side_seam.jpg': (975, 320),
    'stitch_broken_left_sleeve_seam.jpg': (310, 320),
    'stitch_broken_right_side_seam.jpg': (710, 325),
    'stitch_broken_shoulder_seam.jpg': (1020, 100),
    
    # Loose thread (1024x1024)
    'stitch_loose_thread_back_yoke_seam.jpg': (720, 130),
    'stitch_loose_thread_bottom_hem.jpg': (512, 400),
    'stitch_loose_thread_collar_seam.jpg': (425, 120),
    'stitch_loose_thread_cuff_seam.jpg': (780, 890),
    'stitch_loose_thread_right_side_seam.jpg': (600, 380),
    'stitch_loose_thread_shoulder_seam.jpg': (250, 150),
    
    # Open seam (1264x848)
    'stitch_open_seam_back_yoke_seam_1789635312289.jpg': (580, 105),
    'stitch_open_seam_collar_seam_1789635295024.jpg': (640, 100),
    'stitch_open_seam_left_side_seam_1789635243298.jpg': (985, 310),
    'stitch_open_seam_left_sleeve_seam_1789635273374.jpg': (950, 240),
    'stitch_open_seam_right_side_seam_1789635255619.jpg': (710, 310),
    'stitch_open_seam_right_sleeve_seam_1789635284867.jpg': (1120, 240),
    
    # Zigzag (1024x1024)
    'stitch_zigzag_back_yoke_seam.jpg': (300, 115),
    'stitch_zigzag_collar_seam.jpg': (525, 110),
    'stitch_zigzag_left_side_seam.jpg': (770, 320),
    'stitch_zigzag_right_side_seam.jpg': (240, 320),
    'stitch_zigzag_shoulder_seam.jpg': (512, 250),
    'stitch_zigzag_sleeve_seam.jpg': (230, 310),
}

def generate_crops():
    for name, (cx, cy) in CENTERS.items():
        p = os.path.join(STITCH_DIR, name)
        img = cv2.imread(p)
        h, w = img.shape[:2]
        
        # Crop 300x300
        crop_size = 150
        x1 = max(0, cx - crop_size)
        x2 = min(w, cx + crop_size)
        y1 = max(0, cy - crop_size)
        y2 = min(h, cy + crop_size)
        
        crop = img[y1:y2, x1:x2].copy()
        
        # Draw coordinate grid every 25px with global coordinates
        for x in range(x1 - (x1 % 25), x2, 25):
            lx = x - x1
            cv2.line(crop, (lx, 0), (lx, crop.shape[0]), (0, 255, 255), 1)
            cv2.putText(crop, str(x), (lx + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        for y in range(y1 - (y1 % 25), y2, 25):
            ly = y - y1
            cv2.line(crop, (0, ly), (crop.shape[1], ly), (0, 255, 255), 1)
            cv2.putText(crop, str(y), (5, ly - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            
        cv2.putText(crop, f"{name[:25]}", (10, crop.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        out_name = f"crop_{name.replace('.jpg', '')[:30]}.jpg"
        cv2.imwrite(os.path.join(ARTIFACT_DIR, out_name), crop)

generate_crops()
print("All 24 detailed crops created in artifact dir!")
