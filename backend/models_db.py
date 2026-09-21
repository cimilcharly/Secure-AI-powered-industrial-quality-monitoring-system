"""
SQLAlchemy database models for Fabric QC System (Module 11).
Stores users, detections, drift logs, model versions, and configurable cost tables.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text
from backend.database import Base
from src.security.auth import get_password_hash, Role


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default=Role.OPERATOR.value, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String(255), nullable=False)
    defect_class = Column(String(50), nullable=False)  # Damage, Button, Stitch, Color, None, Anomaly
    confidence = Column(Float, default=0.0)
    severity_score = Column(Float, default=0.0)        # 0 - 100
    cost_estimate = Column(Float, default=0.0)         # ₹ INR
    is_anomaly = Column(Boolean, default=False)
    gradcam_path = Column(String(255), nullable=True)
    is_synthetic = Column(Boolean, default=False)
    factory_id = Column(String(50), default="Factory_1", index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class DriftLog(Base):
    __tablename__ = "drift_logs"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    factory_id = Column(String(50), default="Factory_1", index=True)
    drift_score = Column(Float, nullable=False)        # 0.0 - 1.0
    alert_level = Column(String(20), nullable=False)   # NORMAL, WARNING, CRITICAL_DRIFT


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    version_tag = Column(String(50), unique=True, nullable=False)
    trained_on_date = Column(DateTime, default=datetime.utcnow)
    accuracy = Column(Float, default=0.0)
    is_deployed = Column(Boolean, default=False)


class CostConfig(Base):
    __tablename__ = "cost_configs"

    id = Column(Integer, primary_key=True, index=True)
    category = Column(String(50), unique=True, nullable=False)  # Damage, Button, Stitch, Color, Anomaly
    rework_cost = Column(Float, nullable=False)
    scrap_cost = Column(Float, nullable=False)
    recall_risk = Column(Float, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


def seed_defaults(db):
    """Seed initial admin/operator users, initial model version, and cost tables."""
    try:
        # 1. Seed Users
        if db.query(User).count() == 0:
            admin_user = User(
                username="admin",
                password_hash=get_password_hash("admin123"),
                role=Role.ADMIN.value
            )
            operator_user = User(
                username="operator",
                password_hash=get_password_hash("operator123"),
                role=Role.OPERATOR.value
            )
            db.add_all([admin_user, operator_user])

        # 2. Seed Default Model Version
        if db.query(ModelVersion).count() == 0:
            initial_v = ModelVersion(
                version_tag="v1.0.0-dual-mode",
                accuracy=0.884,
                is_deployed=True
            )
            db.add(initial_v)

        # 3. Seed Default Cost Configs in ₹
        if db.query(CostConfig).count() == 0:
            costs = [
                CostConfig(category="Damage", rework_cost=250.0, scrap_cost=950.0, recall_risk=3500.0),
                CostConfig(category="Button", rework_cost=60.0, scrap_cost=300.0, recall_risk=1200.0),
                CostConfig(category="Stitch", rework_cost=120.0, scrap_cost=650.0, recall_risk=2200.0),
                CostConfig(category="Color", rework_cost=180.0, scrap_cost=850.0, recall_risk=2800.0),
                CostConfig(category="Anomaly", rework_cost=300.0, scrap_cost=1100.0, recall_risk=4000.0),
            ]
            db.add_all(costs)

        db.commit()
    except Exception as e:
        db.rollback()
        print(f"[Database Seed] Warning during seeding: {e}")
    finally:
        db.close()
