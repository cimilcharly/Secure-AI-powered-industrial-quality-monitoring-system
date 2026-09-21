"""
Main FastAPI Application Entrypoint for Fabric Defect Inspection System.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from backend.config import APP_TITLE, VERSION, STORAGE_DIR
from backend.database import init_db, DB_TYPE
from backend.routes import auth_router, inspection_router, analytics_router, admin_router

app = FastAPI(
    title=APP_TITLE,
    version=VERSION,
    description="Secure AI-Powered Industrial Quality Monitoring System with YOLOv8m, PatchCore, Grad-CAM, and 3-Node Federated Learning."
)

from starlette.middleware.base import BaseHTTPMiddleware
from src.security.rate_limiter import rate_limit_middleware

# Enable CORS for Streamlit and frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach OWASP Rate Limiting Middleware
app.add_middleware(BaseHTTPMiddleware, dispatch=rate_limit_middleware)

# Mount static storage directory
app.mount("/storage", StaticFiles(directory=STORAGE_DIR), name="storage")

# Include Routers
app.include_router(auth_router)
app.include_router(inspection_router)
app.include_router(analytics_router)
app.include_router(admin_router)

@app.on_event("startup")
def on_startup():
    print(f"[Backend Startup] Initializing Database ({DB_TYPE})...")
    init_db()
    print("[Backend Startup] Initialization complete. System ready for live inspection.")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "database": DB_TYPE,
        "modules": {
            "yolo_detection": "active (dual-mode)",
            "patchcore_anomaly": "active (dual-mode)",
            "severity_scoring": "active (₹ cost lookup)",
            "explainability_gradcam": "active",
            "drift_monitoring": "active",
            "security_aes_sha256": "active",
            "federated_learning": "ready (3-node simulation)"
        }
    }


# Mount frontend web application (index.html, style.css, app.js)
import os
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)

