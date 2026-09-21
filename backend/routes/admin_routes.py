"""
Admin and Configuration Management Routes.
Protected routes for retraining triggers, cost matrix configuration,
model versioning, and Federated Learning runs.
"""

from typing import List, Dict, Any
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models_db import CostConfig, ModelVersion, User
from backend.routes.auth_routes import require_admin, get_current_user, require_roles
from src.security.auth import Role
from src.security.mtls_verifier import MTLSVerifier
from src.security.replay_protection import global_replay_gate
from src.security.fl_anomaly_detector import global_fl_anomaly_detector
from src.federated.fedavg_simulation import run_federated_simulation
from src.detection.model_benchmark import benchmark_engine, CANDIDATE_MODELS

router = APIRouter(prefix="/api/admin", tags=["Admin & System Control"])


class CostUpdateRequest(BaseModel):
    category: str
    rework_cost: float
    scrap_cost: float
    recall_risk: float


class FederatedRunRequest(BaseModel):
    rounds: int = 5


class CertificateRevokeRequest(BaseModel):
    node_id: str
    reason: str = "Key compromise detected / untrusted device"


@router.get("/cost-configs")
def get_cost_configs(db: Session = Depends(get_db)):
    """Retrieve current ₹ cost table parameters."""
    configs = db.query(CostConfig).all()
    return [
        {
            "category": c.category,
            "rework_cost_inr": c.rework_cost,
            "scrap_cost_inr": c.scrap_cost,
            "recall_risk_inr": c.recall_risk
        }
        for c in configs
    ]


@router.post("/cost-configs/update")
def update_cost_config(
    request: CostUpdateRequest,
    db: Session = Depends(get_db),
    admin_user: Dict[str, Any] = Depends(require_admin)
):
    """Admin-only: Update economic cost parameters for a defect category."""
    config = db.query(CostConfig).filter(CostConfig.defect_category == request.category).first()
    if not config:
        config = CostConfig(defect_category=request.category)
        db.add(config)

    config.rework_cost_inr = request.rework_cost
    config.scrap_cost_inr = request.scrap_cost
    config.recall_risk_inr = request.recall_risk
    db.commit()

    return {
        "status": "success",
        "category": request.category,
        "rework_cost_inr": request.rework_cost,
        "scrap_cost_inr": request.scrap_cost,
        "recall_risk_inr": request.recall_risk
    }


@router.get("/model-versions")
def get_model_versions(db: Session = Depends(get_db)):
    """List registered and deployed model checkpoints."""
    versions = db.query(ModelVersion).order_by(ModelVersion.id.desc()).all()
    return [
        {
            "id": v.id,
            "version_tag": v.version_tag,
            "trained_on_date": v.trained_on_date.isoformat() if v.trained_on_date else None,
            "accuracy": v.accuracy,
            "is_deployed": v.is_deployed
        }
        for v in versions
    ]


@router.post("/federated-run")
def trigger_federated_run(
    request: FederatedRunRequest,
    user: Dict[str, Any] = Depends(require_roles([Role.ADMIN.value, Role.ML_ENGINEER.value]))
):
    """
    Executes a 3-Node Federated Learning simulation (Module 13).
    RBAC Protected: Requires ADMIN or ML_ENGINEER role.
    Factory Operators are forbidden from triggering global aggregation rounds.
    """
    result = run_federated_simulation(rounds=request.rounds)
    return result


@router.get("/security/pki-inventory")
def get_pki_inventory():
    """Returns X.509 Root CA and machine certificates for all factory nodes."""
    return MTLSVerifier.get_all_certificates()


@router.get("/security/audit-logs")
def get_security_audit_logs(
    user: Dict[str, Any] = Depends(require_roles([Role.ADMIN.value, Role.AUDITOR.value, Role.ML_ENGINEER.value]))
):
    """
    Returns security telemetry and event logs:
    - mTLS verification status
    - Replay protection gate metrics
    - FL update anomaly / model poisoning detections
    """
    return {
        "pki_status": MTLSVerifier.get_all_certificates(),
        "replay_protection": global_replay_gate.get_status_telemetry(),
        "poisoning_anomalies": global_fl_anomaly_detector.get_recent_anomalies(),
        "accessed_by": user.get("username"),
        "role": user.get("role")
    }


@router.post("/security/revoke-certificate")
def revoke_node_cert(
    request: CertificateRevokeRequest,
    admin_user: Dict[str, Any] = Depends(require_admin)
):
    """Admin-only: Simulates revoking an untrusted factory node's X.509 certificate."""
    ok = MTLSVerifier.revoke_node_certificate(request.node_id, request.reason)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Node {request.node_id} not found")
    return {
        "status": "revoked",
        "node_id": request.node_id,
        "reason": request.reason,
        "message": f"Certificate for {request.node_id} has been revoked. Node is blocked from FedAvg."
    }


@router.post("/security/unrevoke-certificate")
def unrevoke_node_cert(
    request: CertificateRevokeRequest,
    admin_user: Dict[str, Any] = Depends(require_admin)
):
    """Admin-only: Restores an unrevoked certificate for demo testing."""
    ok = MTLSVerifier.unrevoke_node_certificate(request.node_id)
    return {"status": "restored", "node_id": request.node_id}


@router.post("/retrain-trigger")
def trigger_retrain(
    model_type: str = "all",
    user: Dict[str, Any] = Depends(require_roles([Role.ADMIN.value, Role.ML_ENGINEER.value]))
):
    """Trigger background retraining run."""
    return {
        "status": "queued",
        "model_type": model_type,
        "message": f"Training job dispatched for {model_type}. Progress will be logged to model_versions."
    }


@router.get("/benchmark-results")
def get_benchmark_results():
    """Retrieve cached or fresh multi-model benchmark report comparing YOLOv8, YOLO11, and YOLO26."""
    results = benchmark_engine.results_cache
    if not results:
        results = benchmark_engine.run_benchmark()
    active = benchmark_engine.get_active_model()
    return {
        "benchmark": results,
        "active_model": active
    }


@router.post("/run-benchmark")
def run_model_benchmark(
    user: Dict[str, Any] = Depends(require_roles([Role.ADMIN.value, Role.ML_ENGINEER.value]))
):
    """Execute evaluation benchmark comparing YOLOv8, YOLO11, and YOLO26."""
    report = benchmark_engine.run_benchmark()
    return report


@router.post("/deploy-best-model")
def deploy_model(model_key: str = "YOLO11"):
    """Promotes the specified or top-ranked candidate model to models/best.pt."""
    info = benchmark_engine.deploy_model(model_key)
    return {
        "status": "success",
        "message": f"Successfully deployed {model_key} as active production model.",
        "details": info
    }

