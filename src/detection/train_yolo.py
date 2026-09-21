"""
Training script for YOLOv8m Defect Detection (Module 4).
Runs full model training on combined real + synthetic datasets.
Saves exported model weights to models/best.pt for backend inference.
"""

import os
import argparse
import yaml

DEFAULT_DATA_YAML = os.path.join(os.getcwd(), "data", "dataset.yaml")


def train_yolo(
    data_yaml: str = DEFAULT_DATA_YAML,
    model_variant: str = "yolov8m.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    project: str = "runs/train",
    name: str = "fabric_defect_yolov8m"
):
    print(f"=== Starting YOLOv8 Training ===")
    print(f"Data config: {data_yaml}")
    print(f"Base Model:  {model_variant}")
    print(f"Epochs:      {epochs}, Batch: {batch}, Image Size: {imgsz}")

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml}")

    from ultralytics import YOLO

    model = YOLO(model_variant)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        save=True,
        plots=True
    )

    # Save to standard models directory
    best_weights = os.path.join(project, name, "weights", "best.pt")
    target_weights = os.path.join(os.getcwd(), "models", "best.pt")

    if os.path.exists(best_weights):
        os.makedirs(os.path.dirname(target_weights), exist_ok=True)
        import shutil
        shutil.copy2(best_weights, target_weights)
        print(f"Successfully copied trained weights to: {target_weights}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8m on Fabric Defect Dataset")
    parser.add_argument("--data", default=DEFAULT_DATA_YAML, help="Path to data.yaml")
    parser.add_argument("--model", default="yolov8m.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--device", default="cpu", help="Device (cpu or 0, 1...)")
    args = parser.parse_args()

    train_yolo(
        data_yaml=args.data,
        model_variant=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device
    )
