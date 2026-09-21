"""
Cryptographic utilities for Fabric QC System.
Provides AES-256-GCM authenticated encryption/decryption and SHA-256 integrity checks.
"""

import os
import base64
import hashlib
from typing import Tuple, Dict, Any

try:
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    PYCRYPTODOME_AVAILABLE = True
except ImportError:
    PYCRYPTODOME_AVAILABLE = False


# Default system AES key (can be overridden via environment variable)
DEFAULT_AES_KEY_B64 = os.environ.get(
    "FABRIC_AES_KEY",
    "k9r7u3x8z1A4b7e9c2d5f8j1m4p7s0v3y6B9e1h4k7n="
)


def get_default_key() -> bytes:
    """Returns a 32-byte (256-bit) AES key decoded from base64 or generated."""
    try:
        raw = base64.b64decode(DEFAULT_AES_KEY_B64)
        if len(raw) == 32:
            return raw
    except Exception:
        pass
    # Fallback to deterministic 32-byte key
    return hashlib.sha256(b"fabric-qc-secure-industrial-key-2026").digest()


def generate_aes_key() -> str:
    """Generate a random 256-bit AES key as a base64 string."""
    if PYCRYPTODOME_AVAILABLE:
        key = get_random_bytes(32)
    else:
        key = os.urandom(32)
    return base64.b64encode(key).decode('utf-8')


def compute_sha256(data: bytes) -> str:
    """Compute standard SHA-256 hexadecimal digest of byte data."""
    if not isinstance(data, (bytes, bytearray)):
        data = str(data).encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def verify_sha256(data: bytes, expected_hash: str) -> Tuple[bool, str]:
    """
    Verify SHA-256 integrity of data against an expected hash.
    Returns: (is_valid: bool, status_message: str)
    """
    actual_hash = compute_sha256(data)
    if actual_hash.lower() == expected_hash.strip().lower():
        return True, "verified"
    return False, f"tampering detected: hash mismatch (expected {expected_hash[:8]}..., got {actual_hash[:8]}...)"


class AESCipher:
    """AES-256-GCM Authenticated Encryption/Decryption."""

    def __init__(self, key: bytes = None):
        if key is None:
            self.key = get_default_key()
        elif isinstance(key, str):
            self.key = base64.b64decode(key) if len(key) > 32 else hashlib.sha256(key.encode()).digest()
        else:
            self.key = key

        if len(self.key) not in (16, 24, 32):
            self.key = hashlib.sha256(self.key).digest()

    def encrypt(self, plaintext: bytes) -> Dict[str, str]:
        """
        Encrypt bytes using AES-GCM.
        Returns dictionary containing base64 encoded ciphertext, nonce, and auth tag.
        """
        if not isinstance(plaintext, (bytes, bytearray)):
            plaintext = str(plaintext).encode('utf-8')

        nonce = get_random_bytes(12) if PYCRYPTODOME_AVAILABLE else os.urandom(12)
        cipher = AES.new(self.key, AES.MODE_GCM, nonce=nonce)
        ciphertext, tag = cipher.encrypt_and_digest(plaintext)

        return {
            "ciphertext": base64.b64encode(ciphertext).decode('utf-8'),
            "nonce": base64.b64encode(nonce).decode('utf-8'),
            "tag": base64.b64encode(tag).decode('utf-8'),
            "sha256": compute_sha256(plaintext)
        }

    def decrypt(self, encrypted_payload: Dict[str, str]) -> bytes:
        """
        Decrypt payload dict with ciphertext, nonce, and tag.
        Verifies authentication tag and SHA-256 checksum.
        """
        ciphertext = base64.b64decode(encrypted_payload["ciphertext"])
        nonce = base64.b64decode(encrypted_payload["nonce"])
        tag = base64.b64decode(encrypted_payload["tag"])

        cipher = AES.new(self.key, AES.MODE_GCM, nonce=nonce)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)

        if "sha256" in encrypted_payload:
            valid, msg = verify_sha256(plaintext, encrypted_payload["sha256"])
            if not valid:
                raise ValueError(f"Decryption integrity check failed: {msg}")

        return plaintext


def encrypt_payload(data: bytes, key: bytes = None) -> Dict[str, str]:
    """Convenience function to encrypt bytes."""
    cipher = AESCipher(key=key)
    return cipher.encrypt(data)


def decrypt_payload(payload: Dict[str, str], key: bytes = None) -> bytes:
    """Convenience function to decrypt payload."""
    cipher = AESCipher(key=key)
    return cipher.decrypt(payload)
