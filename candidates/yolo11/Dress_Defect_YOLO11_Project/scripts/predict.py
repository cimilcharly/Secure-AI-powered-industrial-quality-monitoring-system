import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--source", required=True)
parser.add_argument("--conf", type=float, default=0.25)
parser.add_argument("--imgsz", type=int, default=640)
args = parser.parse_args()

model = YOLO(args.model)
model.predict(
    source=args.source, conf=args.conf, imgsz=args.imgsz,
    save=True, save_txt=True, save_conf=True,
    project="runs/detect", name="yolo11_predictions"
)
