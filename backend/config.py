"""
Configuration settings for Fabric QC Backend.
"""

import os

APP_TITLE = "Secure Fabric Quality Inspection System API"
VERSION = "1.0.0"

# Secret keys
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "fabric-qc-jwt-super-secret-key-2026-industrial")
FABRIC_AES_KEY = os.environ.get("FABRIC_AES_KEY", "k9r7u3x8z1A4b7e9c2d5f8j1m4p7s0v3y6B9e1h4k7n=")

# Storage paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
UPLOAD_DIR = os.path.join(STORAGE_DIR, "uploads")
GRADCAM_DIR = os.path.join(STORAGE_DIR, "gradcam")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)

# Database URL (MySQL with SQLite automatic fallback)
MYSQL_USER = os.environ.get("MYSQL_USER", "fabric_admin")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD", "fabric_pass2026")
MYSQL_HOST = os.environ.get("MYSQL_HOST", "localhost")
MYSQL_PORT = os.environ.get("MYSQL_PORT", "3306")
MYSQL_DB = os.environ.get("MYSQL_DB", "fabric_qc_db")

MYSQL_DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
SQLITE_DATABASE_URL = f"sqlite:///{os.path.join(BASE_DIR, 'fabric_qc.db')}"
