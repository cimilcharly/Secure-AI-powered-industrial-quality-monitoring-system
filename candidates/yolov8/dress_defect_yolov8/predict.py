from ultralytics import YOLO
import argparse
import os
import csv
import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, help="Path to trained best.pt")
    p.add_argument("--source", required=True, help="Image or folder of test images")
    p.add_argument("--output", default="runs/predict")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    args = p.parse_args()

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        save_txt=True,
        save_conf=True,
        project=args.output,
        name="results",
        exist_ok=True
    )

    os.makedirs(args.output, exist_ok=True)
    csv_path = os.path.join(args.output, "predictions.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "class_id", "class_name", "confidence",
                         "x1", "y1", "x2", "y2", "width", "height"])

        for r in results:
            names = r.names
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = [float(x) for x in box.xyxy[0].tolist()]
                writer.writerow([
                    os.path.basename(r.path), cls, names[cls], round(conf, 6),
                    round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2),
                    round(x2-x1, 2), round(y2-y1, 2)
                ])

    print(f"\nAnnotated images: {args.output}/results/")
    print(f"YOLO text labels: {args.output}/results/labels/")
    print(f"Prediction CSV:   {csv_path}")

if __name__ == "__main__":
    main()
