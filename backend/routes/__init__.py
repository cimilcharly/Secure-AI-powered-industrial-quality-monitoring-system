from .auth_routes import router as auth_router
from .inspection_routes import router as inspection_router
from .analytics_routes import router as analytics_router
from .admin_routes import router as admin_router

__all__ = [
    "auth_router",
    "inspection_router",
    "analytics_router",
    "admin_router"
]
