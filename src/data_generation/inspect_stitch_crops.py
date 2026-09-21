import cv2
import numpy as np
import os
import glob

ARTIFACT_DIR = r"C:\Users\HP\.gemini\antigravity-ide\brain\7c5aecdd-ac14-4e26-a4ad-68e0b6140d38"
STITCH_DIR = r"c:\Users\HP\Desktop\Dress Defect\data\ai_generated\stitch"

# Subclasses and their 6 images
subclasses = ["broken", "loose_thread", "open_seam", "zigzag"]

for sub in subclasses:
    imgs = sorted(glob.glob(os.path.join(STITCH_DIR, f"*{sub}*.jpg")))
    print(f"Subclass {sub}: {len(imgs)} images")
    
    # We will create a contact sheet: 2 rows of 3 images, downscaled to 512x... with a grid
    tiles = []
    for p in imgs:
        img = cv2.imread(p)
        h, w = img.shape[:2]
        
        # Overlay a subtle 100px grid and text labels with coordinates
        display = img.copy()
        for x in range(0, w, 100):
            cv2.line(display, (x, 0), (x, h), (180, 180, 180), 1)
            cv2.putText(display, str(x), (x + 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        for y in range(0, h, 100):
            cv2.line(display, (0, y), (w, y), (180, 180, 180), 1)
            cv2.putText(display, str(y), (5, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
        # Draw filename
        fname = os.path.basename(p).replace(".jpg", "").replace("stitch_", "").replace("1789635", "")
        cv2.rectangle(display, (0, h - 35), (w, h), (0, 0, 0), -1)
        cv2.putText(display, f"{fname} ({w}x{h})", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Resize to 480x360 or 400x400 for sheet
        tile = cv2.resize(display, (400, 400))
        tiles.append(tile)
        
    # Stack 2 rows of 3
    if len(tiles) >= 6:
        row1 = np.hstack(tiles[0:3])
        row2 = np.hstack(tiles[3:6])
        sheet = np.vstack([row1, row2])
        out_path = os.path.join(ARTIFACT_DIR, f"stitch_{sub}_grid.jpg")
        cv2.imwrite(out_path, sheet)
        print(f"Saved {out_path}")

print("Inspection grids generated!")
