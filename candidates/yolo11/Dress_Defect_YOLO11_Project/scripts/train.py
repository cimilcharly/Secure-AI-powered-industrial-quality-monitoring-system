import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--device", default="")
args = parser.parse_args()

model = YOLO("yolo11n.pt")
kwargs = dict(
    data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
    project="runs/detect", name="yolo11_dress_defect", pretrained=True,
    seed=42, patience=30, plots=True
)
if args.device:
    kwargs["device"] = args.device
model.train(**kwargs)
