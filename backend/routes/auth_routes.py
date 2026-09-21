"""
Authentication and Authorization endpoints.
Implements RFC 7519 JWT lifecycle, token rotation, revocation,
and fine-grained RBAC dependencies.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models_db import User
from src.security.auth import (
    create_access_token,
    create_refresh_token,
    verify_token,
    revoke_token,
    verify_password,
    verify_factory_access,
    DEFAULT_USERS,
    Role,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])
security = HTTPBearer(auto_error=False)


class LoginRequest(BaseModel):
    username: str
    password: str


class DemoLoginRequest(BaseModel):
    persona: str  # 'admin', 'operator_tirupur', 'operator_coimbatore', 'ml_engineer', 'auditor'


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    username: str
    role: str
    factory_id: Optional[str] = None
    display_name: str
    expires_in_seconds: int
    auth_mode: str = "DEMO_SIMULATED_OIDC"


def get_current_token_payload(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """Extracts and verifies JWT claims from Bearer header."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: Missing Bearer token in Authorization header",
            headers={"WWW-Authenticate": "Bearer"}
        )
    token = credentials.credentials
    payload = verify_token(token, expected_type="access")
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid, expired, or revoked authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return payload


def get_current_user(
    payload: Dict[str, Any] = Depends(get_current_token_payload),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Returns current user details from validated JWT payload."""
    username = payload.get("sub")
    # Check demo users store first, then DB
    user = DEFAULT_USERS.get(username)
    if user:
        return {
            "username": user["username"],
            "role": user["role"].value if isinstance(user["role"], Role) else user["role"],
            "factory_id": user.get("factory_id"),
            "display_name": user.get("display_name", user["username"])
        }

    db_user = db.query(User).filter(User.username == username).first()
    if not db_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User identity no longer exists in authority directory"
        )
    return {
        "username": db_user.username,
        "role": db_user.role,
        "factory_id": None,
        "display_name": db_user.username
    }


def require_roles(allowed_roles: List[str]):
    """Factory dependency for RBAC role authorization."""
    def role_checker(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        user_role = user.get("role")
        if user_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: Insufficient privileges. Required role in {allowed_roles}, your role is '{user_role}'"
            )
        return user
    return role_checker


def require_admin(user: Dict[str, Any] = Depends(require_roles([Role.ADMIN.value]))) -> Dict[str, Any]:
    return user


def require_ml_or_admin(user: Dict[str, Any] = Depends(require_roles([Role.ADMIN.value, Role.ML_ENGINEER.value]))) -> Dict[str, Any]:
    return user


@router.post("/demo-login", response_model=TokenResponse)
def demo_login(request: DemoLoginRequest):
    """
    Demo Authentication Mode:
    Simulates Enterprise OIDC SSO Provider with pre-configured personas.
    Issues RFC 7519 compliant signed JWT access token + refresh token.
    """
    persona_key = request.persona.lower().strip()
    user = DEFAULT_USERS.get(persona_key)
    if not user:
        available = list(DEFAULT_USERS.keys())
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid demo persona '{request.persona}'. Available personas: {available}"
        )

    role_val = user["role"].value if isinstance(user["role"], Role) else user["role"]

    access_token = create_access_token({
        "sub": user["username"],
        "role": role_val,
        "factory_id": user.get("factory_id"),
        "display_name": user["display_name"]
    })
    refresh_token = create_refresh_token(user["username"])

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "username": user["username"],
        "role": role_val,
        "factory_id": user.get("factory_id"),
        "display_name": user["display_name"],
        "expires_in_seconds": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "auth_mode": "DEMO_SIMULATED_OIDC"
    }


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate with username and password, returns signed JWT & refresh token."""
    user = DEFAULT_USERS.get(request.username)
    if user and verify_password(request.password, user["password_hash"]):
        role_val = user["role"].value if isinstance(user["role"], Role) else user["role"]
        access_token = create_access_token({
            "sub": user["username"],
            "role": role_val,
            "factory_id": user.get("factory_id"),
            "display_name": user["display_name"]
        })
        refresh_token = create_refresh_token(user["username"])
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "username": user["username"],
            "role": role_val,
            "factory_id": user.get("factory_id"),
            "display_name": user["display_name"],
            "expires_in_seconds": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "auth_mode": "ENTERPRISE_PASSWORD_AUTH"
        }

    db_user = db.query(User).filter(User.username == request.username).first()
    if not db_user or not verify_password(request.password, db_user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    access_token = create_access_token({
        "sub": db_user.username,
        "role": db_user.role,
        "factory_id": None,
        "display_name": db_user.username
    })
    refresh_token = create_refresh_token(db_user.username)

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "username": db_user.username,
        "role": db_user.role,
        "factory_id": None,
        "display_name": db_user.username,
        "expires_in_seconds": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "auth_mode": "DATABASE_CREDENTIALS"
    }


@router.post("/refresh", response_model=TokenResponse)
def refresh_token(request: RefreshRequest):
    """
    RFC 6749 Token Rotation:
    Consumes a valid refresh token, revokes it, and issues a fresh access & refresh token pair.
    """
    payload = verify_token(request.refresh_token, expected_type="refresh")
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )

    # Revoke old refresh token (Token Rotation defense against token theft)
    revoke_token(request.refresh_token)

    username = payload.get("sub")
    user = DEFAULT_USERS.get(username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    role_val = user["role"].value if isinstance(user["role"], Role) else user["role"]
    new_access = create_access_token({
        "sub": user["username"],
        "role": role_val,
        "factory_id": user.get("factory_id"),
        "display_name": user["display_name"]
    })
    new_refresh = create_refresh_token(user["username"])

    return {
        "access_token": new_access,
        "refresh_token": new_refresh,
        "token_type": "bearer",
        "username": user["username"],
        "role": role_val,
        "factory_id": user.get("factory_id"),
        "display_name": user["display_name"],
        "expires_in_seconds": ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        "auth_mode": "TOKEN_ROTATION"
    }


@router.post("/logout")
def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Revokes the presenting access token immediately."""
    if credentials:
        revoke_token(credentials.credentials)
    return {"status": "logged_out", "message": "Access token revoked successfully"}


@router.get("/me")
def get_profile(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Return authenticated user profile and active scopes."""
    return current_user

