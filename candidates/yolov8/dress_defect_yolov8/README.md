# Dress Defect Detection — YOLOv8

This project trains a YOLOv8 object-detection model on dress/clothing defect images
and produces bounding-box predictions plus Precision, Recall, F1, mAP50 and mAP50-95.

## 1. Dataset format

Put your images and LabelImg YOLO `.txt` labels here:

dataset/
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/

For each image, the label file has one line per object:

class_id x_center y_center width height

Coordinates are normalized to 0–1, which is the YOLO format.

Example:
0 0.512 0.431 0.120 0.085

Edit `dataset/data.yaml` and replace the `names` section with your actual defect
classes. Example:

names:
  0: hole
  1: stain
  2: tear
  3: loose_thread

## 2. Install

Windows PowerShell:

python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

If you have an NVIDIA GPU, install a compatible PyTorch/CUDA build if needed.

## 3. Train

Example:

python train.py --model yolov8s.pt --epochs 100 --imgsz 640 --batch 16 --device 0

CPU:

python train.py --model yolov8n.pt --epochs 50 --imgsz 640 --batch 4 --device cpu

The trained model will normally be at:

runs/dress_defect/weights/best.pt

Training plots and metrics are also saved under `runs/`.

## 4. Predict test images

After training:

python predict.py --weights runs/dress_defect/weights/best.pt --source dataset/images/test --conf 0.25 --device 0

Outputs:
- Annotated images with YOLO bounding boxes
- YOLO `.txt` prediction labels
- `predictions.csv` containing class, confidence and pixel bounding boxes

## 5. Calculate Precision / Recall / F1 / mAP

For a labeled test set:

python evaluate.py --weights runs/dress_defect/weights/best.pt --data dataset/data.yaml --split test --device 0

This produces:
- Precision
- Recall
- F1
- mAP@0.50
- mAP@0.50:0.95
- plots/confusion matrix and related evaluation artifacts

IMPORTANT:
A test image must have a matching ground-truth LabelImg `.txt` file for
Precision/Recall/F1/mAP evaluation. If you upload only unlabeled images, the
model can predict boxes and confidence scores, but meaningful evaluation
metrics cannot be computed.

## 6. Dashboard integration

`api.py` provides an optional FastAPI endpoint:

GET  /health
POST /predict

Run:

pip install fastapi uvicorn python-multipart
uvicorn api:app --host 0.0.0.0 --port 8000

Set the trained model if it is in another location:

Windows PowerShell:
$env:MODEL_PATH="runs/dress_defect/weights/best.pt"

Your dashboard can POST an image to `/predict` and receive JSON containing:
class_id, class_name, confidence, and `[x1,y1,x2,y2]` pixel coordinates.

## Recommended project workflow

1. Generate/collect defect images.
2. Label defect bounding boxes in LabelImg using YOLO format.
3. Split into train/val/test (for example 70/20/10).
4. Keep the test set untouched until final evaluation.
5. Train YOLOv8.
6. Run evaluation on labeled test data.
7. Run `predict.py` on new/unlabeled images for deployment.
8. Connect `api.py` to your existing dashboard.

Do not use generated images in both train and test if they are near-duplicates;
this can inflate reported metrics.
