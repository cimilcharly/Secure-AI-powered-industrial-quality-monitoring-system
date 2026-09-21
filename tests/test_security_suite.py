"""
Comprehensive Security & Defense Verification Suite.
Tests:
  1. RFC 7519 JWT Standard Claims (iss, aud, sub, exp, nbf)
  2. Token Refresh, Rotation, and JTI Blacklist Revocation
  3. Machine mTLS Certificate Validation & Revocation (RFC 8705)
  4. Anti-Replay Protection Gate (Packet Freshness & Duplicate ID Rejection)
  5. OWASP Broken Object Level Authorization (BOLA) Defense
  6. Federated Learning Update Anomaly & Poisoning Outlier Detection
"""
import os
import sys
import time
import unittest
import numpy as np

# Ensure workspace root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.security.auth import (
    create_access_token,
    create_refresh_token,
    verify_token,
    revoke_token,
    verify_factory_access,
    Role
)
from src.security.mtls_verifier import MTLSVerifier
from src.security.replay_protection import ReplayProtectionGate, ReplayAttackError
from src.security.fl_anomaly_detector import FLAnomalyDetector


class TestSecuritySuite(unittest.TestCase):

    def setUp(self):
        self.username = "test_operator"
        self.factory_id = "Factory_2"

    # --- 1. JWT Standard Claims & Lifecycle ---
    def test_jwt_standard_claims(self):
        token = create_access_token({
            "sub": self.username,
            "role": Role.FACTORY_OPERATOR.value,
            "factory_id": self.factory_id
        }, expires_delta_minutes=15)

        payload = verify_token(token, expected_type="access")
        self.assertIsNotNone(payload, "Token should be valid")
        self.assertEqual(payload["iss"], "fabricqc-auth-authority")
        self.assertEqual(payload["aud"], "fabricqc-api")
        self.assertEqual(payload["sub"], self.username)
        self.assertEqual(payload["role"], Role.FACTORY_OPERATOR.value)
        self.assertEqual(payload["factory_id"], self.factory_id)
        self.assertIn("jti", payload)
        self.assertIn("exp", payload)
        self.assertIn("nbf", payload)

    def test_token_revocation(self):
        token = create_access_token({"sub": "revoked_user", "role": "admin"})
        self.assertIsNotNone(verify_token(token))

        # Revoke token
        revoked = revoke_token(token)
        self.assertTrue(revoked)

        # Re-verification must fail
        self.assertIsNone(verify_token(token), "Revoked token must not verify")

    def test_refresh_token_rotation(self):
        refresh = create_refresh_token("operator_tirupur")
        payload = verify_token(refresh, expected_type="refresh")
        self.assertIsNotNone(payload)
        self.assertEqual(payload["token_type"], "refresh")

        # Refresh token must not be accepted as an access token
        self.assertIsNone(verify_token(refresh, expected_type="access"))

    # --- 2. Machine mTLS Certificate Handshake ---
    def test_mtls_valid_handshake(self):
        for node in ["Factory_1", "Factory_2", "Factory_3"]:
            ok, msg, telemetry = MTLSVerifier.verify_node_handshake(node)
            self.assertTrue(ok, f"Node {node} should pass mTLS handshake")
            self.assertEqual(telemetry["status"], "MUTUAL_TLS_AUTHENTICATED")

    def test_mtls_revoked_node_rejection(self):
        node = "Factory_3"
        MTLSVerifier.revoke_node_certificate(node, reason="Compromised hardware probe")
        ok, msg, _ = MTLSVerifier.verify_node_handshake(node)
        self.assertFalse(ok, "Revoked certificate node must be rejected")
        self.assertIn("REVOKED", msg)

        # Restore for subsequent tests
        MTLSVerifier.unrevoke_node_certificate(node)
        ok_restored, _, _ = MTLSVerifier.verify_node_handshake(node)
        self.assertTrue(ok_restored)

    # --- 3. Replay Protection Gate ---
    def test_replay_duplicate_rejection(self):
        gate = ReplayProtectionGate(time_tolerance_seconds=300)
        pkt_id = "pkt_tirupur_r1_test001"
        ts = time.time()

        # First delivery: Accepted
        ok, msg = gate.validate_packet("Factory_2", 1, pkt_id, ts, current_active_round=1)
        self.assertTrue(ok)

        # Second delivery (replay attempt): Must raise ReplayAttackError
        with self.assertRaises(ReplayAttackError):
            gate.validate_packet("Factory_2", 1, pkt_id, ts, current_active_round=1)

    def test_replay_stale_timestamp_rejection(self):
        gate = ReplayProtectionGate(time_tolerance_seconds=300)
        stale_ts = time.time() - 900  # 15 minutes ago
        ok, msg = gate.validate_packet("Factory_1", 1, "pkt_stale", stale_ts)
        self.assertFalse(ok)
        self.assertIn("Timestamp out of bounds", msg)

    # --- 4. OWASP BOLA Object-Level Authorization ---
    def test_bola_operator_factory_scoping(self):
        tirupur_operator = {
            "sub": "operator_tirupur",
            "role": Role.FACTORY_OPERATOR.value,
            "factory_id": "Factory_2"
        }

        # Authorized to access Factory_2
        self.assertTrue(verify_factory_access(tirupur_operator, "Factory_2"))

        # Blocked from accessing Factory_1 (BOLA Defense)
        self.assertFalse(verify_factory_access(tirupur_operator, "Factory_1"))

        # Admin has cross-factory clearance
        admin_user = {"sub": "admin", "role": Role.ADMIN.value, "factory_id": None}
        self.assertTrue(verify_factory_access(admin_user, "Factory_1"))
        self.assertTrue(verify_factory_access(admin_user, "Factory_2"))

    # --- 5. FL Update Anomaly & Poisoning Detection ---
    def test_fl_poisoning_anomaly_detection(self):
        detector = FLAnomalyDetector(multiplier_threshold=3.0, max_absolute_norm=6.0)
        global_w = np.zeros((4, 64))

        # 3 Honest factories with small normal gradients
        updates = {
            "Factory_1": np.random.normal(0, 0.05, size=(4, 64)),
            "Factory_2": np.random.normal(0, 0.05, size=(4, 64)),
            "Factory_3": np.random.normal(0, 0.05, size=(4, 64)),
            # 1 Compromised/Malicious node injecting large poisoned weights
            "Malicious_Node": np.ones((4, 64)) * 12.0
        }

        accepted, telemetry = detector.inspect_updates(updates, global_w, round_id=1)

        # Honest nodes accepted
        self.assertIn("Factory_1", accepted)
        self.assertIn("Factory_2", accepted)
        self.assertIn("Factory_3", accepted)

        # Poisoned node excluded from FedAvg
        self.assertNotIn("Malicious_Node", accepted)
        malicious_record = [t for t in telemetry if t["node_id"] == "Malicious_Node"][0]
        self.assertTrue(malicious_record["is_anomaly"])
        self.assertEqual(malicious_record["status"], "FLAGGED_POISONING_ATTACK")


if __name__ == "__main__":
    unittest.main()
