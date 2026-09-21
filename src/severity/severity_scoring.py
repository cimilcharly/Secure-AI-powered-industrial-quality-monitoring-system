"""
Severity Scoring and Financial Cost-Impact Estimation Module (Module 6).
Computes defect area relative to garment, combines with confidence & defect type weighting,
determines severity band (Low 0-30, Medium 31-70, High 71-100),
and estimates repair/scrap/recall cost impact in Indian Rupees (₹).
"""

from enum import Enum
from typing import Dict, Any, List, Optional


class SeverityLevel(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


# Category baseline weights (Damage is highest risk of structural scrap)
DEFAULT_CATEGORY_WEIGHTS = {
    "Damage": 1.00,
    "Button": 0.55,
    "Stitch": 0.70,
    "Color": 0.60
}

# Configurable Indian Rupee (₹) unit costs table
DEFAULT_COST_TABLE = {
    "Damage": {
        "rework_cost": 250.0,   # Patching/darning
        "scrap_cost": 950.0,    # Total scrap cost of fabric
        "recall_risk": 3500.0   # If delivered defective to consumer/buyer
    },
    "Button": {
        "rework_cost": 60.0,    # Re-stitching button
        "scrap_cost": 300.0,
        "recall_risk": 1200.0
    },
    "Stitch": {
        "rework_cost": 120.0,   # Seam re-run
        "scrap_cost": 650.0,
        "recall_risk": 2200.0
    },
    "Color": {
        "rework_cost": 180.0,   # Spot dye/wash
        "scrap_cost": 850.0,
        "recall_risk": 2800.0
    },
    "Anomaly": {
        "rework_cost": 300.0,   # Unclassified anomaly investigation
        "scrap_cost": 1100.0,
        "recall_risk": 4000.0
    }
}


class SeverityScorer:
    """Computes industrial severity and financial risk in ₹."""

    def __init__(
        self,
        cost_table: Optional[Dict[str, Dict[str, float]]] = None,
        category_weights: Optional[Dict[str, float]] = None
    ):
        self.cost_table = cost_table or DEFAULT_COST_TABLE
        self.category_weights = category_weights or DEFAULT_CATEGORY_WEIGHTS

    def compute_single_defect(
        self,
        class_name: str,
        confidence: float,
        area_ratio: float,
        is_anomaly: bool = False
    ) -> Dict[str, Any]:
        """
        Compute severity score (0 - 100) and cost impact for an individual defect.
        Formula:
          area_component = min(1.0, area_ratio * 15.0)  # Area > ~6.7% maxes area component
          conf_component = confidence
          weight = category_weights.get(class_name, 0.6)
          score = (0.50 * area_component + 0.30 * conf_component + 0.20 * weight) * 100
        """
        weight = self.category_weights.get(class_name, 0.60)
        if is_anomaly:
            weight = max(weight, 0.85)

        # Scale area ratio so small defects (e.g. 1%) still get realistic proportional weight
        area_component = min(1.0, area_ratio * 12.0)
        conf_component = max(0.0, min(1.0, confidence))

        raw_score = (0.50 * area_component + 0.30 * conf_component + 0.20 * weight) * 100.0
        score = round(max(0.0, min(100.0, raw_score)), 1)

        # Classify into severity bands
        if score <= 30.0:
            band = SeverityLevel.LOW
            action = "Rework"
        elif score <= 70.0:
            band = SeverityLevel.MEDIUM
            action = "Rework / Supervisory Inspection"
        else:
            band = SeverityLevel.HIGH
            action = "Scrap / Reject Garment"

        # Calculate ₹ financial impact
        costs = self.cost_table.get(class_name, self.cost_table.get("Anomaly", {}))
        rework_cost = costs.get("rework_cost", 150.0)
        scrap_cost = costs.get("scrap_cost", 800.0)
        recall_risk = costs.get("recall_risk", 2500.0)

        if band == SeverityLevel.LOW:
            est_cost = rework_cost
        elif band == SeverityLevel.MEDIUM:
            est_cost = rework_cost * 1.5
        else:
            est_cost = scrap_cost

        return {
            "severity_score": score,
            "severity_band": band.value,
            "recommended_action": action,
            "area_ratio_pct": round(area_ratio * 100, 2),
            "estimated_cost_inr": round(est_cost, 2),
            "cost_breakdown": {
                "rework_inr": rework_cost,
                "scrap_inr": scrap_cost,
                "recall_risk_inr": recall_risk
            }
        }

    def evaluate_inspection(
        self,
        detections: List[Dict[str, Any]],
        anomaly_score: float = 0.0,
        is_anomaly: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluates overall garment inspection across multiple defect detections and anomaly score.
        """
        evaluated_defects = []
        total_cost_inr = 0.0
        max_severity = 0.0

        for det in detections:
            res = self.compute_single_defect(
                class_name=det.get("class_name", "Damage"),
                confidence=det.get("confidence", 0.8),
                area_ratio=det.get("area_ratio", 0.02),
                is_anomaly=False
            )
            # Merge into detection dict
            merged = {**det, **res}
            evaluated_defects.append(merged)
            total_cost_inr += res["estimated_cost_inr"]
            if res["severity_score"] > max_severity:
                max_severity = res["severity_score"]

        # If unseen anomaly detected without known YOLO boxes
        if is_anomaly and not detections:
            res = self.compute_single_defect(
                class_name="Anomaly",
                confidence=anomaly_score,
                area_ratio=0.03,
                is_anomaly=True
            )
            evaluated_defects.append({
                "class_id": -1,
                "class_name": "Unseen Anomaly",
                "confidence": round(anomaly_score, 3),
                "box": [0, 0, 0, 0],
                **res
            })
            total_cost_inr += res["estimated_cost_inr"]
            max_severity = max(max_severity, res["severity_score"])

        # Overall garment grade
        if max_severity <= 30.0:
            overall_band = SeverityLevel.LOW
        elif max_severity <= 70.0:
            overall_band = SeverityLevel.MEDIUM
        else:
            overall_band = SeverityLevel.HIGH

        return {
            "evaluated_detections": evaluated_defects,
            "overall_severity_score": round(max_severity, 1),
            "overall_severity_band": overall_band.value,
            "total_estimated_cost_inr": round(total_cost_inr, 2),
            "defect_count": len(evaluated_defects)
        }


# Global instance
_default_scorer = SeverityScorer()


def calculate_severity_and_cost(
    detections: List[Dict[str, Any]],
    anomaly_score: float = 0.0,
    is_anomaly: bool = False
) -> Dict[str, Any]:
    """Helper function to calculate severity and cost."""
    return _default_scorer.evaluate_inspection(
        detections=detections,
        anomaly_score=anomaly_score,
        is_anomaly=is_anomaly
    )
