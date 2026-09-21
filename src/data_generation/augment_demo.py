import cv2
import numpy as np
import os
import random

def augment_image(image):
    h, w = image.shape[:2]
    
    # Random brightness and contrast
    alpha = random.uniform(0.8, 1.2) # Contrast control
    beta = random.randint(-20, 20)   # Brightness control
    augmented = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    # Random slight rotation (-5 to 5 degrees)
    angle = random.uniform(-5, 5)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    augmented = cv2.warpAffine(augmented, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Random subtle Gaussian Noise (Proper float32 clipping to prevent uint8 underflow)
    if random.random() > 0.5:
        noise = np.random.normal(0, 4.0, augmented.shape).astype(np.float32)
        augmented = np.clip(augmented.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
    return augmented

def main():
    source_img_path = r"c:\Users\HP\Desktop\Dress Defect\data\ai_generated\stitch\stitch_broken_left_side_seam.jpg"
    out_dir = r"c:\Users\HP\Desktop\Dress Defect\data\demo_augmented"
    
    if not os.path.exists(source_img_path):
        print(f"Error: Could not find source image at {source_img_path}")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    
    print("Loading base image...")
    base_img = cv2.imread(source_img_path)
    
    print(f"Generating 10 augmented variations...")
    for i in range(1, 11):
        aug_img = augment_image(base_img)
        out_path = os.path.join(out_dir, f"aug_broken_stitch_{i}.jpg")
        cv2.imwrite(out_path, aug_img)
        print(f"Saved: {out_path}")
        
    print("Done!")

if __name__ == "__main__":
    main()
