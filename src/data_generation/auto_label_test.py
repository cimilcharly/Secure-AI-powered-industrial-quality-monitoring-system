import cv2
import numpy as np

image_path = r"C:\Users\HP\.gemini\antigravity-ide\brain\7c5aecdd-ac14-4e26-a4ad-68e0b6140d38\.user_uploaded\media_1789633883007.png"
img = cv2.imread(image_path)
h, w, _ = img.shape

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Run Canny edge detector
edges = cv2.Canny(gray, 50, 150)

# The defect is on the left sleeve, so let's isolate a region of interest
# x from 0.15*w to 0.35*w, y from 0.5*h to 0.7*h
roi_x1 = int(0.15 * w)
roi_x2 = int(0.35 * w)
roi_y1 = int(0.50 * h)
roi_y2 = int(0.70 * h)

# Mask everything outside ROI
mask = np.zeros_like(edges)
mask[roi_y1:roi_y2, roi_x1:roi_x2] = 255
edges_roi = cv2.bitwise_and(edges, mask)

# Find contours of edges in the ROI
contours, _ = cv2.findContours(edges_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh > 5:
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + bw)
            y_max = max(y_max, y + bh)
            
    if x_max > 0:
        class_id = 2 # Stitch
        
        # Calculate YOLO format
        x_center = (x_min + x_max) / 2.0 / w
        y_center = (y_min + y_max) / 2.0 / h
        box_width = (x_max - x_min) / w
        box_height = (y_max - y_min) / h
        
        # Add padding
        box_width *= 1.2
        box_height *= 1.2
        
        print("Detected YOLO Label:")
        print(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
        
        cv2.rectangle(img, (int(x_center*w - box_width*w/2), int(y_center*h - box_height*h/2)), 
                           (int(x_center*w + box_width*w/2), int(y_center*h + box_height*h/2)), (0, 0, 255), 2)
        out_path = r"C:\Users\HP\.gemini\antigravity-ide\brain\7c5aecdd-ac14-4e26-a4ad-68e0b6140d38\labeled_sample.jpg"
        cv2.imwrite(out_path, img)
        print(f"Saved visualization to {out_path}")
    else:
        print("No valid edges found in ROI.")
else:
    print("No edges found in ROI.")
