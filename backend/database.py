"""
Database engine and session management with MySQL and SQLite dual-mode support.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from backend.config import MYSQL_DATABASE_URL, SQLITE_DATABASE_URL

Base = declarative_base()

# Attempt connection to MySQL; if unavailable, fallback to SQLite
ENGINE = None
DB_TYPE = "unknown"

try:
    # Try MySQL first if configured
    mysql_engine = create_engine(MYSQL_DATABASE_URL, pool_pre_ping=True, connect_args={"connect_timeout": 3})
    with mysql_engine.connect() as conn:
        ENGINE = mysql_engine
        DB_TYPE = "MySQL"
        print("[Database] Successfully connected to MySQL production database.")
except Exception as e:
    # Graceful fallback to SQLite
    sqlite_engine = create_engine(SQLITE_DATABASE_URL, connect_args={"check_same_thread": False})
    ENGINE = sqlite_engine
    DB_TYPE = "SQLite (Local Dual-Mode Fallback)"
    print(f"[Database] MySQL not reachable ({e}). Using local SQLite database.")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=ENGINE)


def get_db():
    """FastAPI dependency for DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initializes tables and default seed data."""
    from backend import models_db
    Base.metadata.create_all(bind=ENGINE)
    models_db.seed_defaults(SessionLocal())
