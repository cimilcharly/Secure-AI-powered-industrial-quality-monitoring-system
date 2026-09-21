import cv2
import numpy as np
import os
import csv
import glob

# Constants
DATA_DIR = r"c:\Users\HP\Desktop\Dress Defect\data\ai_generated\stitch"
DEBUG_DIR = os.path.join(DATA_DIR, "debug")
CSV_PATH = r"c:\Users\HP\Desktop\Dress Defect\src\data_generation\prompts.csv"
CLASS_ID = 2  # Stitch

# ROI Definitions (normalized coordinates: x1, y1, x2, y2)
ROI_MAP = {
    "left side seam": (0.20, 0.40, 0.50, 0.90),
    "right side seam": (0.50, 0.40, 0.80, 0.90),
    "left sleeve seam": (0.05, 0.40, 0.40, 0.80),
    "right sleeve seam": (0.60, 0.40, 0.95, 0.80),
    "shoulder seam": (0.20, 0.10, 0.50, 0.40), 
    "bottom hem": (0.25, 0.70, 0.75, 0.95),
    "back yoke seam": (0.25, 0.10, 0.75, 0.40),
    "collar seam": (0.35, 0.05, 0.65, 0.25),
    "cuff seam": (0.65, 0.60, 0.95, 0.90), 
    "sleeve seam": (0.05, 0.40, 0.40, 0.80), 
}

def auto_label_image(img_path, position_str):
    if not os.path.exists(img_path):
        return False
        
    img = cv2.imread(img_path)
    if img is None:
        return False
        
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Mild blur to remove basic fabric texture noise before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    
    # Get ROI
    roi_coords = None
    for k in ROI_MAP:
        if k in position_str.lower():
            roi_coords = ROI_MAP[k]
            break
            
    if roi_coords is None:
        print(f"Warning: No ROI mapping found for {position_str}")
        return False
        
    roi_x1 = int(roi_coords[0] * w)
    roi_y1 = int(roi_coords[1] * h)
    roi_x2 = int(roi_coords[2] * w)
    roi_y2 = int(roi_coords[3] * h)
    
    mask = np.zeros_like(edges)
    mask[roi_y1:roi_y2, roi_x1:roi_x2] = 255
    edges_roi = cv2.bitwise_and(edges, mask)
    
    contours, _ = cv2.findContours(edges_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback to generous fixed bounding box if Canny finds nothing
        print(f"  Fallback needed for {os.path.basename(img_path)} - No edges found in ROI")
        x_min, y_min, x_max, y_max = roi_x1, roi_y1, roi_x2, roi_y2
    else:
        x_min, y_min = w, h
        x_max, y_max = 0, 0
        valid_contours = 0
        
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bw * bh > 50: # ignore small noise, focus only on main defect
                valid_contours += 1
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + bw)
                y_max = max(y_max, y + bh)
                
        if valid_contours == 0:
            print(f"  Fallback needed for {os.path.basename(img_path)} - Contours too small")
            x_min, y_min, x_max, y_max = roi_x1, roi_y1, roi_x2, roi_y2
    
    # Convert to YOLO
    # Add padding (15%)
    box_w = x_max - x_min
    box_h = y_max - y_min
    
    if box_w < 10 or box_h < 10:
        # If it's incredibly tiny, default to a generous size around center of ROI
        center_x = (roi_x1 + roi_x2) / 2
        center_y = (roi_y1 + roi_y2) / 2
        x_min = int(center_x - w*0.05)
        x_max = int(center_x + w*0.05)
        y_min = int(center_y - h*0.05)
        y_max = int(center_y + h*0.05)
        box_w = x_max - x_min
        box_h = y_max - y_min
    
    pad_x = 0
    pad_y = 0
    
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(w, x_max + pad_x)
    y_max = min(h, y_max + pad_y)
    
    # YOLO format calculation
    yolo_x_center = ((x_min + x_max) / 2.0) / w
    yolo_y_center = ((y_min + y_max) / 2.0) / h
    yolo_w = (x_max - x_min) / float(w)
    yolo_h = (y_max - y_min) / float(h)
    
    # Save .txt label
    txt_filename = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
    txt_path = os.path.join(DATA_DIR, txt_filename)
    
    with open(txt_path, "w") as f:
        f.write(f"{CLASS_ID} {yolo_x_center:.6f} {yolo_y_center:.6f} {yolo_w:.6f} {yolo_h:.6f}\n")
        
    # Save debug image
    debug_img = img.copy()
    cv2.rectangle(debug_img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 3) # Red bounding box
    cv2.rectangle(debug_img, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 2) # Blue ROI
    
    os.makedirs(DEBUG_DIR, exist_ok=True)
    debug_path = os.path.join(DEBUG_DIR, os.path.basename(img_path))
    cv2.imwrite(debug_path, debug_img)
    
    return True

def main():
    if not os.path.exists(CSV_PATH):
        print("CSV not found.")
        return
        
    os.makedirs(DEBUG_DIR, exist_ok=True)
    
    # Read prompts
    rows = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            
    success_count = 0
    
    for row in rows:
        # check if file actually exists (either .png or .jpg or with a timestamp)
        base_name = row['image_filename'].split('.')[0]
        # glob to find it (since some have timestamps like _1789635295024)
        search_pattern = os.path.join(DATA_DIR, f"{base_name}*.*")
        matches = glob.glob(search_pattern)
        
        for match in matches:
            if match.endswith(('.png', '.jpg', '.jpeg')):
                print(f"Labeling {os.path.basename(match)}...")
                if auto_label_image(match, row['position']):
                    success_count += 1
                break
                
    print(f"\nDone! Labeled {success_count} images.")
    print(f"Check debug images in {DEBUG_DIR}")

if __name__ == "__main__":
    main()
