"""
Dedicated YOLOv8 Training Script for Cimil's Stitch Class (Class 2 / Single-Class Detector).
Trains a single-class detector on data/stitch_dataset/dataset.yaml.
Saves the optimized weights to models/stitch_best.pt.
"""

import os
import argparse
import shutil
from ultralytics import YOLO

DEFAULT_DATA_YAML = r"c:\Users\HP\Desktop\Dress Defect\data\stitch_dataset\dataset.yaml"
OUTPUT_WEIGHTS = r"c:\Users\HP\Desktop\Dress Defect\models\stitch_best.pt"

def train_stitch_model(
    epochs: int = 20,
    batch: int = 8,
    imgsz: int = 640,
    model_name: str = "yolov8n.pt",
    device: str = "cpu"
):
    print("=" * 65)
    print("STARTING YOLOV8 TRAINING FOR CIMIL'S STITCH CLASS")
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
        project="runs/stitch_train",
        name="stitch_detector",
        exist_ok=True,
        save=True,
        plots=True,
        patience=10
    )

    # Check and copy best weights
    best_pt = os.path.join("runs", "stitch_train", "stitch_detector", "weights", "best.pt")
    if os.path.exists(best_pt):
        os.makedirs(os.path.dirname(OUTPUT_WEIGHTS), exist_ok=True)
        shutil.copy2(best_pt, OUTPUT_WEIGHTS)
        print(f"\n[OK] Training completed! Best model saved to: {OUTPUT_WEIGHTS}")
    else:
        print("\nWarning: best.pt not found in run directory.")

    # Run independent validation on the test set
    print("\n" + "=" * 65)
    print("RUNNING INDEPENDENT EVALUATION ON TEST SET (20 IMAGES)")
    print("=" * 65)
    test_metrics = model.val(data=DEFAULT_DATA_YAML, split="test", imgsz=imgsz, device=device)
    print("\nTest Evaluation Results:")
    print(f"  mAP@50:    {test_metrics.box.map50 * 100:.2f}%")
    print(f"  mAP@50-95: {test_metrics.box.map * 100:.2f}%")
    print(f"  Precision: {test_metrics.box.p[0] * 100:.2f}%" if len(test_metrics.box.p) > 0 else "  Precision: N/A")
    print(f"  Recall:    {test_metrics.box.r[0] * 100:.2f}%" if len(test_metrics.box.r) > 0 else "  Recall: N/A")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stitch YOLOv8 Model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", default="cpu", help="Device ('cpu' or GPU index '0')")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained base model weights")
    args = parser.parse_args()

    train_stitch_model(
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        model_name=args.model
    )
