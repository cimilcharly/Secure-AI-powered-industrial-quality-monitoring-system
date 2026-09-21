"""
Dedicated YOLOv8 Training Script for Bharath's Damage Class (Class 0).
Trains a single-class detector on data/damage_dataset/dataset.yaml.
Saves the optimized weights to models/damage_best.pt.
"""

import os
import argparse
import shutil
from ultralytics import YOLO

DEFAULT_DATA_YAML = r"c:\Users\HP\Desktop\Dress Defect\data\damage_dataset\dataset.yaml"
OUTPUT_WEIGHTS = r"c:\Users\HP\Desktop\Dress Defect\models\damage_best.pt"

def train_damage_model(
    epochs: int = 20,
    batch: int = 8,
    imgsz: int = 640,
    model_name: str = "yolov8n.pt",
    device: str = "cpu"
):
    print("=" * 65)
    print("STARTING YOLOV8 TRAINING FOR BHARATH'S DAMAGE CLASS")
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
        project="runs/damage_train",
        name="damage_detector",
        exist_ok=True,
        save=True,
        plots=True,
        patience=10
    )

    # Check and copy best weights
    best_pt = os.path.join("runs", "damage_train", "damage_detector", "weights", "best.pt")
    if os.path.exists(best_pt):
        os.makedirs(os.path.dirname(OUTPUT_WEIGHTS), exist_ok=True)
        shutil.copy2(best_pt, OUTPUT_WEIGHTS)
        print(f"\n[OK] Training completed! Best model saved to: {OUTPUT_WEIGHTS}")
    else:
        print("\nWarning: best.pt not found in run directory.")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Damage YOLOv8 Model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", default="cpu", help="Device ('cpu' or GPU index '0')")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained base model weights")
    args = parser.parse_args()

    train_damage_model(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        model_name=args.model
    )
