import cv2
import numpy as np
import os
import glob

ARTIFACT_DIR = r"C:\Users\HP\.gemini\antigravity-ide\brain\7c5aecdd-ac14-4e26-a4ad-68e0b6140d38"
STITCH_DIR = r"c:\Users\HP\Desktop\Dress Defect\data\ai_generated\stitch"

subclasses = ["broken", "loose_thread", "open_seam", "zigzag"]

for sub in subclasses:
    imgs = sorted(glob.glob(os.path.join(STITCH_DIR, f"*{sub}*.jpg")))
    
    # Let's create an annotated full-resolution image for each image in this subclass
    # Stack 3 rows of 2 images
    grid_rows = []
    for i in range(0, 6, 2):
        pair = []
        for j in range(2):
            idx = i + j
            p = imgs[idx]
            img = cv2.imread(p)
            h, w = img.shape[:2]
            
            # Standardize display to 600x400
            disp = cv2.resize(img, (600, 400))
            
            # Draw coordinate grid: every 50px in original coordinates, mapped to display
            # disp_w = 600, disp_h = 400
            sx = 600.0 / w
            sy = 400.0 / h
            
            for ox in range(0, w, 100):
                dx = int(ox * sx)
                cv2.line(disp, (dx, 0), (dx, 400), (0, 255, 255), 1)
                cv2.putText(disp, str(ox), (dx + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
            for oy in range(0, h, 100):
                dy = int(oy * sy)
                cv2.line(disp, (0, dy), (600, dy), (0, 255, 255), 1)
                cv2.putText(disp, str(oy), (5, dy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                
            fname = os.path.basename(p).replace(".jpg", "").replace("stitch_", "").replace("1789635", "")
            cv2.rectangle(disp, (0, 375), (600, 400), (0, 0, 0), -1)
            cv2.putText(disp, f"{fname} [{w}x{h}]", (10, 393), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            pair.append(disp)
        grid_rows.append(np.hstack(pair))
    
    full_cat = np.vstack(grid_rows)
    out_p = os.path.join(ARTIFACT_DIR, f"cat_annotated_{sub}.jpg")
    cv2.imwrite(out_p, full_cat)
    print(f"Saved {out_p}")
