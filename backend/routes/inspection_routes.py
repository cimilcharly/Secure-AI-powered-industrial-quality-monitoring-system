"""
Inspection and AI Inference Routes (Modules 3, 4, 5, 6, 7, 8, 9, 11).
Orchestrates the entire fabric defect inspection pipeline.
"""

import os
import uuid
import base64
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import cv2
import numpy as np
from typing import Optional, Dict, Any

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from backend.database import get_db
from backend.config import UPLOAD_DIR, GRADCAM_DIR
from backend.models_db import Detection, DriftLog
from backend.routes.auth_routes import get_current_user
from src.security.auth import verify_token, verify_factory_access
from src.security.crypto_utils import compute_sha256, verify_sha256, AESCipher
from src.preprocessing.preprocessor import preprocess_image, image_to_bytes
from src.detection.yolo_detector import detect_defects
from src.anomaly_detection.patchcore_engine import detect_anomaly
from src.severity.severity_scoring import calculate_severity_and_cost
from src.explainability.gradcam import explain_detection
from src.drift.drift_monitor import compute_drift_score

router = APIRouter(prefix="/api/inspection", tags=["Inspection & AI Pipelines"])
security = HTTPBearer(auto_error=False)


@router.post("/upload-and-inspect")
async def upload_and_inspect(
    file: UploadFile = File(...),
    factory_id: str = Form("Factory_1"),
    model_name: str = Form("auto"),
    sha256_checksum: str = Form(None),
    is_encrypted_payload: bool = Form(False),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
):
    """
    Main industrial inspection endpoint.
    Executes the full pipeline:
      0. OWASP BOLA Object-level authorization check (factory_id scoping)
      1. AES payload decryption (if encrypted)
      2. SHA-256 data integrity verification
      3. Preprocessing (standardize, CLAHE, denoise)
      4. YOLOv8m detection & classification (Damage / Stitch / Ensemble)
      5. PatchCore anomaly detection
      6. Severity scoring & ₹ cost calculation
      7. Grad-CAM heatmap generation & localization recall
      8. Drift score calculation
      9. DB logging
    """
    # 0. OWASP Broken Object Level Authorization (BOLA) Check
    if credentials:
        payload = verify_token(credentials.credentials, expected_type="access")
        if payload and not verify_factory_access(payload, factory_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden (BOLA Defense): Operator '{payload.get('sub')}' is assigned to {payload.get('factory_id')}, cannot inspect or submit to '{factory_id}'."
            )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 1. AES Decryption (if encrypted payload was uploaded)
    if is_encrypted_payload:
        try:
            cipher = AESCipher()
            # If payload was JSON with nonce/ciphertext/tag
            import json
            payload_dict = json.loads(raw_bytes.decode('utf-8'))
            raw_bytes = cipher.decrypt(payload_dict)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"AES Decryption failed: {str(e)}")

    # 2. SHA-256 Integrity Verification
    integrity_status = "verified"
    if sha256_checksum:
        is_valid, msg = verify_sha256(raw_bytes, sha256_checksum)
        if not is_valid:
            integrity_status = msg
            raise HTTPException(status_code=400, detail=f"Data integrity failed: {msg}")

    # 3. Preprocessing
    processed_bgr, normalized_float, preproc_meta = preprocess_image(raw_bytes)

    # Save original/preprocessed image
    img_id = f"{uuid.uuid4().hex[:10]}_{int(datetime.utcnow().timestamp())}"
    raw_filename = f"{img_id}_raw.jpg"
    raw_path = os.path.join(UPLOAD_DIR, raw_filename)
    cv2.imwrite(raw_path, processed_bgr)

    # 4. YOLOv8 Defect Detection
    det_result = detect_defects(processed_bgr, return_annotated=True, model_name=model_name)
    detections = det_result["detections"]
    annotated_bgr = det_result["annotated_image"]

    # 5. PatchCore Anomaly Detection
    anomaly_result = detect_anomaly(processed_bgr, return_heatmap=True)
    anomaly_score = anomaly_result["anomaly_score"]
    is_anomaly = anomaly_result["is_anomaly"]

    # 6. Severity Scoring & ₹ Cost Estimation
    severity_result = calculate_severity_and_cost(
        detections=detections,
        anomaly_score=anomaly_score,
        is_anomaly=is_anomaly
    )

    # 7. Grad-CAM Explainability & Localization Recall
    boxes = [d["box"] for d in detections]
    gradcam_bgr, raw_heatmap, recall_metrics = explain_detection(processed_bgr, boxes)

    gradcam_filename = f"{img_id}_gradcam.jpg"
    gradcam_path = os.path.join(GRADCAM_DIR, gradcam_filename)
    cv2.imwrite(gradcam_path, gradcam_bgr)

    # 8. Drift Monitoring
    drift_result = compute_drift_score(processed_bgr, factory_id=factory_id)

    # 9. Database Logging
    # Determine primary defect category
    if detections:
        primary_class = detections[0]["class_name"]
        primary_conf = detections[0]["confidence"]
    elif is_anomaly:
        primary_class = "Anomaly"
        primary_conf = anomaly_score
    else:
        primary_class = "Defect-Free"
        primary_conf = 1.0 - anomaly_score

    db_detection = Detection(
        image_path=raw_filename,
        defect_class=primary_class,
        confidence=float(primary_conf),
        severity_score=float(severity_result["overall_severity_score"]),
        cost_estimate=float(severity_result["total_estimated_cost_inr"]),
        is_anomaly=bool(is_anomaly),
        gradcam_path=gradcam_filename,
        is_synthetic=False,
        factory_id=factory_id,
        timestamp=datetime.utcnow()
    )
    db.add(db_detection)

    db_drift = DriftLog(
        timestamp=datetime.utcnow(),
        factory_id=factory_id,
        drift_score=float(drift_result["drift_score"]),
        alert_level=str(drift_result["alert_level"])
    )
    db.add(db_drift)
    db.commit()
    db.refresh(db_detection)

    # Base64 representations for quick UI display without separate HTTP calls
    _, ann_buf = cv2.imencode('.jpg', annotated_bgr)
    _, gcam_buf = cv2.imencode('.jpg', gradcam_bgr)

    return {
        "inspection_id": db_detection.id,
        "factory_id": factory_id,
        "integrity": integrity_status,
        "primary_defect_class": primary_class,
        "confidence": primary_conf,
        "defect_count": len(detections),
        "detections": severity_result["evaluated_detections"],
        "anomaly": {
            "is_anomaly": is_anomaly,
            "anomaly_score": anomaly_score,
            "threshold": anomaly_result["threshold"]
        },
        "severity": {
            "overall_score": severity_result["overall_severity_score"],
            "severity_band": severity_result["overall_severity_band"],
            "total_cost_inr": severity_result["total_estimated_cost_inr"]
        },
        "explainability": {
            "localization_recall_metrics": recall_metrics,
            "gradcam_image_path": f"/api/inspection/image/gradcam/{gradcam_filename}"
        },
        "drift": {
            "drift_score": drift_result["drift_score"],
            "alert_level": drift_result["alert_level"],
            "is_drift_detected": drift_result["is_drift_detected"]
        },
        "images_base64": {
            "annotated": base64.b64encode(ann_buf).decode('utf-8'),
            "gradcam": base64.b64encode(gcam_buf).decode('utf-8')
        }
    }


@router.get("/image/upload/{filename}")
def get_upload_image(filename: str):
    """Serve uploaded raw inspection image."""
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/image/gradcam/{filename}")
def get_gradcam_image(filename: str):
    """Serve Grad-CAM explanation image."""
    path = os.path.join(GRADCAM_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Grad-CAM image not found")
    return FileResponse(path, media_type="image/jpeg")
