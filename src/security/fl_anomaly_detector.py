"""
Federated Learning Update Anomaly and Poisoning Detection Engine.
Detects abnormal gradient/weight magnitudes from compromised factory nodes
by calculating update L2 norms and statistical cohort dispersion.
Thwarts model poisoning, backdoor injection, and Byzantine failure modes.
"""

import numpy as np
from typing import Dict, Any, List, Tuple


class FLAnomalyDetector:
    """
    Evaluates incoming model updates before FedAvg aggregation.
    Filters out poisoned or diverging model weights.
    """

    def __init__(self, multiplier_threshold: float = 3.0, max_absolute_norm: float = 8.0):
        self.multiplier_threshold = multiplier_threshold
        self.max_absolute_norm = max_absolute_norm
        self.anomaly_history: List[Dict[str, Any]] = []

    def inspect_updates(
        self,
        local_updates: Dict[str, np.ndarray],
        global_weights: np.ndarray,
        round_id: int
    ) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
        """
        Calculates L2 update norms and filters out malicious/poisoned outliers.
        Returns: (accepted_updates_dict, audit_telemetry_list)
        """
        norms: Dict[str, float] = {}
        delta_weights: Dict[str, np.ndarray] = {}

        # 1. Compute L2 norm of delta for each factory node
        for node_id, w_k in local_updates.items():
            delta = w_k - global_weights
            norm_val = float(np.linalg.norm(delta))
            norms[node_id] = round(norm_val, 4)
            delta_weights[node_id] = delta

        # 2. Calculate cohort baseline
        norm_values = list(norms.values())
        if not norm_values:
            return {}, []

        median_norm = float(np.median(norm_values))
        cutoff_threshold = max(self.max_absolute_norm, median_norm * self.multiplier_threshold)

        accepted_updates: Dict[str, np.ndarray] = {}
        telemetry: List[Dict[str, Any]] = []

        # 3. Classify each node update
        for node_id, w_k in local_updates.items():
            norm_val = norms[node_id]
            is_anomaly = norm_val > cutoff_threshold

            record = {
                "round": round_id,
                "node_id": node_id,
                "update_norm": norm_val,
                "cohort_median_norm": round(median_norm, 4),
                "threshold": round(cutoff_threshold, 4),
                "is_anomaly": is_anomaly,
                "status": "FLAGGED_POISONING_ATTACK" if is_anomaly else "ACCEPTED_BENIGN"
            }

            if is_anomaly:
                record["action"] = "EXCLUDED_FROM_AGGREGATION"
                self.anomaly_history.append(record)
            else:
                record["action"] = "INCLUDED_IN_FEDAVG"
                accepted_updates[node_id] = w_k

            telemetry.append(record)

        return accepted_updates, telemetry

    def get_recent_anomalies(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Returns log of detected poisoning attempts."""
        return self.anomaly_history[-limit:]


# Global Anomaly Detector instance
global_fl_anomaly_detector = FLAnomalyDetector()
