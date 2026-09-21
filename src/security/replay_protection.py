"""
Replay Protection and Monotonic Sequence Engine.
Protects Federated Learning updates and sensitive inspection uploads against replay attacks.
Validates packet freshness (±300s window), unique packet IDs, and round monotonicity.
"""

import time
from typing import Dict, Any, Tuple, Optional


class ReplayAttackError(Exception):
    """Raised when an identical packet or expired timestamp is detected."""
    pass


class ReplayProtectionGate:
    """
    Guards the Federated Aggregator and API Gateway against duplicate or stale packet replay.
    """

    def __init__(self, time_tolerance_seconds: int = 300):
        self.tolerance = time_tolerance_seconds
        # Maps packet_id -> dict(timestamp, node_id, round_id)
        self._seen_packets: Dict[str, Dict[str, Any]] = {}
        self.stats = {
            "verified_count": 0,
            "replays_thwarted": 0,
            "stale_timestamp_count": 0
        }

    def _prune_expired_cache(self, now: float):
        """Removes entries older than 2x the tolerance window."""
        cutoff = now - (self.tolerance * 2)
        expired_keys = [
            pid for pid, meta in self._seen_packets.items()
            if meta["received_at"] < cutoff
        ]
        for pid in expired_keys:
            del self._seen_packets[pid]

    def validate_packet(
        self,
        node_id: str,
        round_id: int,
        packet_id: str,
        timestamp: float,
        current_active_round: Optional[int] = None
    ) -> Tuple[bool, str]:
        """
        Validates the freshness and uniqueness of an encrypted update packet.
        Returns: (is_valid: bool, status_message: str)
        """
        now = time.time()
        self._prune_expired_cache(now)

        # 1. Round Monotonicity Check
        if current_active_round is not None and round_id != current_active_round:
            return False, f"Invalid round ID: packet for round {round_id} received, active round is {current_active_round}"

        # 2. Timestamp Freshness Check (Sliding Clock Window)
        skew = abs(now - timestamp)
        if skew > self.tolerance:
            self.stats["stale_timestamp_count"] += 1
            return False, f"Timestamp out of bounds: skew is {round(skew, 1)}s (max allowable is {self.tolerance}s)"

        # 3. Duplicate Packet ID Check (Replay Detection)
        if packet_id in self._seen_packets:
            self.stats["replays_thwarted"] += 1
            original = self._seen_packets[packet_id]
            raise ReplayAttackError(
                f"REPLAY_ATTACK_DETECTED: Packet ID '{packet_id[:12]}...' was previously consumed "
                f"from {original['node_id']} for round {original['round_id']} at t={int(original['received_at'])}."
            )

        # Record packet
        self._seen_packets[packet_id] = {
            "node_id": node_id,
            "round_id": round_id,
            "packet_timestamp": timestamp,
            "received_at": now
        }
        self.stats["verified_count"] += 1
        return True, "Packet Freshness & Anti-Replay Verified"

    def get_status_telemetry(self) -> Dict[str, Any]:
        """Returns live anti-replay telemetry for dashboard."""
        return {
            "active_cache_size": len(self._seen_packets),
            "total_verified": self.stats["verified_count"],
            "replays_blocked": self.stats["replays_thwarted"],
            "stale_blocked": self.stats["stale_timestamp_count"],
            "tolerance_window_seconds": self.tolerance,
            "gate_status": "ACTIVE_ENFORCEMENT"
        }


# Global Replay Gate instance
global_replay_gate = ReplayProtectionGate(time_tolerance_seconds=300)
