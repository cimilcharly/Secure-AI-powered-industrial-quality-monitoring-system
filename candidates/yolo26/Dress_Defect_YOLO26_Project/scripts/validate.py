import argparse
from ultralytics import YOLO
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--output", default="results.csv")
args = parser.parse_args()

model = YOLO(args.model)
r = model.val(data=args.data, imgsz=args.imgsz, batch=args.batch, plots=True, verbose=False)
p, rec = float(r.box.mp), float(r.box.mr)
f1 = 2*p*rec/(p+rec) if p+rec else 0
df = pd.DataFrame([{
    "model": Path(args.model).stem,
    "precision": p, "recall": rec, "f1": f1,
    "mAP50": float(r.box.map50), "mAP50-95": float(r.box.map)
}])
df.to_csv(args.output, index=False)
print(df.to_string(index=False))
