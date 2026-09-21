"""
Dedicated YOLOv8 Training Script for Dhakshina's Button Defect Class (Class 1).
Trains a single-class detector on data/button_dataset/dataset.yaml.
Saves the optimized weights to models/button_best.pt.
"""

import os
import argparse
import shutil
from ultralytics import YOLO

DEFAULT_DATA_YAML = r"c:\Users\HP\Desktop\Dress Defect\data\button_dataset\dataset.yaml"
OUTPUT_WEIGHTS = r"c:\Users\HP\Desktop\Dress Defect\models\button_best.pt"

def train_button_model(
    epochs: int = 20,
    batch: int = 8,
    imgsz: int = 640,
    model_name: str = "yolov8n.pt",
    device: str = "cpu"
):
    print("=" * 65)
    print("STARTING YOLOV8 TRAINING FOR DHAKSHINA'S BUTTON DEFECT CLASS")
    print(f"Data YAML: {DEFAULT_DATA_YAML}")
    print(f"Base Model: {model_name}")
    print(f"Epochs: {epochs} | Batch: {batch} | Image Size: {imgsz} | Device: {device}")
    print("=" * 65)

    model = YOLO(model_name)
    results = model.train(
        data=DEFAULT_DATA_YAML,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project="runs/button_train",
        name="button_detector",
        exist_ok=True,
        save=True,
        plots=True,
        patience=10
    )

    # Check and copy best weights (Ultralytics may place under runs/detect/ or runs/)
    candidate_best = [
        os.path.join("runs", "detect", "runs", "button_train", "button_detector", "weights", "best.pt"),
        os.path.join("runs", "button_train", "button_detector", "weights", "best.pt"),
    ]
    candidate_last = [
        os.path.join("runs", "detect", "runs", "button_train", "button_detector", "weights", "last.pt"),
        os.path.join("runs", "button_train", "button_detector", "weights", "last.pt"),
    ]
    
    saved = False
    for pt in candidate_best:
        if os.path.exists(pt):
            os.makedirs(os.path.dirname(OUTPUT_WEIGHTS), exist_ok=True)
            shutil.copy2(pt, OUTPUT_WEIGHTS)
            print(f"\n[OK] Training completed! Best model saved to: {OUTPUT_WEIGHTS}")
            saved = True
            break
            
    if not saved:
        for pt in candidate_last:
            if os.path.exists(pt):
                os.makedirs(os.path.dirname(OUTPUT_WEIGHTS), exist_ok=True)
                shutil.copy2(pt, OUTPUT_WEIGHTS)
                print(f"\n[OK] Training completed! Last model saved to: {OUTPUT_WEIGHTS}")
                saved = True
                break
        if not saved:
            print("\nWarning: weights not found in run directory.")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Button YOLOv8 Model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", default="cpu", help="Device ('cpu' or GPU index '0')")
    parser.add_argument("--base", default="yolov8n.pt", help="Base pretrained model")
    args = parser.parse_args()

    train_button_model(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        model_name=args.base,
        device=args.device
    )
