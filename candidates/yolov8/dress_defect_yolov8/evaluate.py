from ultralytics import YOLO
import argparse
import os
import json

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--data", default="dataset/data.yaml")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--output", default="runs/evaluation")
    args = p.parse_args()

    model = YOLO(args.weights)
    m = model.val(
        data=args.data, split=args.split, imgsz=args.imgsz,
        device=args.device, plots=True, project=args.output,
        name="metrics", exist_ok=True
    )

    precision = float(m.box.mp)
    recall = float(m.box.mr)
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0.0)

    report = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mAP50": float(m.box.map50),
        "mAP50-95": float(m.box.map),
        "split": args.split
    }

    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nPer-class metrics/plots are in: {args.output}/metrics/")

if __name__ == "__main__":
    main()
