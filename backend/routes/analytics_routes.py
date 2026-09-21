"""
Analytics and Historical Telemetry Routes.
Provides historical detection logs, drift monitoring trends, and KPI statistics
with multi-factory filtering.
"""

from typing import Optional, List
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from backend.database import get_db
from backend.models_db import Detection, DriftLog

router = APIRouter(prefix="/api/analytics", tags=["Analytics & Telemetry"])


@router.get("/summary")
def get_analytics_summary(
    factory_id: Optional[str] = Query(None, description="Filter by Factory Node"),
    db: Session = Depends(get_db)
):
    """Returns high-level KPI metrics for the monitoring dashboard."""
    query = db.query(Detection)
    if factory_id and factory_id != "All":
        query = query.filter(Detection.factory_id == factory_id)

    total_inspections = query.count()
    if total_inspections == 0:
        return {
            "total_inspected": 0,
            "defect_count": 0,
            "defect_rate_pct": 0.0,
            "total_cost_impact_inr": 0.0,
            "class_distribution": {},
            "severity_breakdown": {"Low": 0, "Medium": 0, "High": 0}
        }

    defective_query = query.filter(Detection.defect_class != "Defect-Free")
    defect_count = defective_query.count()
    defect_rate = round((defect_count / total_inspections) * 100.0, 2)

    total_cost = query.with_entities(func.sum(Detection.cost_estimate)).scalar() or 0.0

    # Class distribution
    class_counts = (
        query.with_entities(Detection.defect_class, func.count(Detection.id))
        .group_by(Detection.defect_class)
        .all()
    )
    class_distribution = {c[0]: c[1] for c in class_counts}

    # Severity bands
    low_count = query.filter(Detection.severity_score <= 30.0).count()
    med_count = query.filter(Detection.severity_score > 30.0, Detection.severity_score <= 70.0).count()
    high_count = query.filter(Detection.severity_score > 70.0).count()

    return {
        "total_inspected": total_inspections,
        "defect_count": defect_count,
        "defect_rate_pct": defect_rate,
        "total_cost_impact_inr": round(float(total_cost), 2),
        "class_distribution": class_distribution,
        "severity_breakdown": {
            "Low": low_count,
            "Medium": med_count,
            "High": high_count
        }
    }


@router.get("/history")
def get_detection_history(
    factory_id: Optional[str] = Query(None),
    defect_class: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Retrieve historical defect inspection records."""
    query = db.query(Detection)
    if factory_id and factory_id != "All":
        query = query.filter(Detection.factory_id == factory_id)
    if defect_class and defect_class != "All":
        query = query.filter(Detection.defect_class == defect_class)

    records = query.order_by(desc(Detection.timestamp)).limit(limit).all()

    return [
        {
            "id": r.id,
            "image_path": r.image_path,
            "defect_class": r.defect_class,
            "confidence": round(r.confidence, 3),
            "severity_score": round(r.severity_score, 1),
            "cost_estimate_inr": round(r.cost_estimate, 2),
            "is_anomaly": r.is_anomaly,
            "gradcam_path": r.gradcam_path,
            "factory_id": r.factory_id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None
        }
        for r in records
    ]


@router.get("/drift")
def get_drift_telemetry(
    factory_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Retrieve rolling drift scores and alert levels."""
    query = db.query(DriftLog)
    if factory_id and factory_id != "All":
        query = query.filter(DriftLog.factory_id == factory_id)

    records = query.order_by(desc(DriftLog.timestamp)).limit(limit).all()

    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "factory_id": r.factory_id,
            "drift_score": round(r.drift_score, 4),
            "alert_level": r.alert_level
        }
        for r in reversed(records)
    ]
