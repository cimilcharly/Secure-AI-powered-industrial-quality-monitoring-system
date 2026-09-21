# Optional FastAPI service for connecting your existing dashboard.
# Install: pip install fastapi uvicorn python-multipart
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import tempfile, os

app = FastAPI(title="Dress Defect YOLO API")
MODEL_PATH = os.environ.get("MODEL_PATH", "runs/dress_defect/weights/best.pt")
model = YOLO(MODEL_PATH)

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        r = model.predict(source=path, conf=0.25, verbose=False)[0]
        detections = []
        if r.boxes is not None:
            for b in r.boxes:
                cls = int(b.cls.item())
                detections.append({
                    "class_id": cls,
                    "class_name": r.names[cls],
                    "confidence": float(b.conf.item()),
                    "box": [float(x) for x in b.xyxy[0].tolist()]
                })
        return JSONResponse({"filename": file.filename, "detections": detections})
    finally:
        os.unlink(path)
