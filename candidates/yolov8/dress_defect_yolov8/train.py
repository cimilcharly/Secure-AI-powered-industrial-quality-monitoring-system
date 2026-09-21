from ultralytics import YOLO
import argparse
import os

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset/data.yaml")
    p.add_argument("--model", default="yolov8n.pt",
                   help="yolov8n.pt, yolov8s.pt, yolov8m.pt, etc.")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0",
                   help="0 for NVIDIA GPU, cpu for CPU")
    p.add_argument("--project", default="runs")
    p.add_argument("--name", default="dress_defect")
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=True,
        plots=True,
        save=True
    )

    # Final validation: precision, recall, mAP50 and mAP50-95
    metrics = model.val(data=args.data, imgsz=args.imgsz, device=args.device)
    print("\n=== VALIDATION METRICS ===")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    if metrics.box.mp + metrics.box.mr > 0:
        f1 = 2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr)
        print(f"F1:        {f1:.4f}")

if __name__ == "__main__":
    main()
