# Dress Defect Detection - YOLO11

Standalone YOLO11 project for training a dress-defect object detector with YOLO-format labels.

## Dataset
Create:
dataset/
  images/train
  images/val
  images/test
  labels/train
  labels/val
  labels/test

Copy `dataset/data.yaml.example` to `dataset/data.yaml` and set your class names.

## Install
pip install -r requirements.txt

## Train
python scripts/train.py --data dataset/data.yaml --epochs 100 --imgsz 640 --batch 16

For a small GPU, use --batch 4 or --batch 8.

## Validate
python scripts/validate.py --data dataset/data.yaml --model runs/detect/yolo11_dress_defect/weights/best.pt

Outputs:
Precision, Recall, F1, mAP50, mAP50-95.

## Predict with bounding boxes
python scripts/predict.py --model runs/detect/yolo11_dress_defect/weights/best.pt --source dataset/images/test

Prediction images are saved under runs/detect/yolo11_predictions/.

## Fair comparison
For comparison with another model, use exactly the same:
- dataset split
- classes
- image size
- epochs
- batch size
- seed
- test images

Pretrained weights are downloaded automatically by Ultralytics.
