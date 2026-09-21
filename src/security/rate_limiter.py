"""
Sliding-Window Rate Limiting Engine.
Protects against brute-force attacks on authentication endpoints
and resource exhaustion on AI inference and federated learning jobs.
Conforms to OWASP API Security Top 10 recommendations (API4: Unrestricted Resource Consumption).
"""

import time
from collections import defaultdict
from typing import Dict, List, Tuple
from fastapi import Request, HTTPException, status


class SlidingWindowRateLimiter:
    """In-memory sliding window rate limiter with per-route thresholds."""

    def __init__(self):
        # Maps client_key -> route_prefix -> list of request timestamps
        self._history: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

        # Default rules: (max_requests, window_seconds)
        self.rules: Dict[str, Tuple[int, int]] = {
            "/api/auth/login": (5, 60),          # 5 attempts / min (brute force defense)
            "/api/auth/demo-login": (15, 60),    # 15 demo switches / min
            "/api/inspection/upload": (30, 60),  # 30 inspection requests / min
            "/api/admin/federated-run": (10, 60), # 10 FL trigger rounds / min
            "default": (120, 60)                # 120 general requests / min
        }

    def _get_client_id(self, request: Request) -> str:
        """Extracts client IP or authenticated subject from headers."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        client = request.client
        return client.host if client else "127.0.0.1"

    def _match_rule(self, path: str) -> Tuple[int, int]:
        """Finds matching rate limit rule for the path."""
        for route_prefix, rule in self.rules.items():
            if route_prefix != "default" and path.startswith(route_prefix):
                return rule
        return self.rules["default"]

    def check_rate_limit(self, request: Request) -> Tuple[bool, int, int]:
        """
        Checks if current request is allowed under sliding window.
        Returns: (is_allowed, remaining_requests, reset_seconds)
        """
        client_id = self._get_client_id(request)
        path = request.url.path
        max_requests, window_seconds = self._match_rule(path)

        now = time.time()
        window_start = now - window_seconds

        # Prune older entries
        timestamps = self._history[client_id][path]
        valid_timestamps = [t for t in timestamps if t > window_start]
        self._history[client_id][path] = valid_timestamps

        if len(valid_timestamps) >= max_requests:
            earliest = valid_timestamps[0]
            reset_seconds = max(1, int(window_seconds - (now - earliest)))
            return False, 0, reset_seconds

        # Record this request
        self._history[client_id][path].append(now)
        remaining = max_requests - (len(valid_timestamps) + 1)
        return True, remaining, window_seconds


# Global instance
rate_limiter = SlidingWindowRateLimiter()


async def rate_limit_middleware(request: Request, call_next):
    """FastAPI middleware applying rate limits and returning rate limit headers."""
    # Skip static files and docs
    path = request.url.path
    if (
        path.startswith("/frontend") or
        path.startswith("/assets") or
        path.startswith("/docs") or
        path.startswith("/openapi.json") or
        path.startswith("/health") or
        path == "/" or
        path == "/style.css" or
        path == "/app.js" or
        path == "/three_engine.js"
    ):
        return await call_next(request)

    allowed, remaining, reset_secs = rate_limiter.check_rate_limit(request)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too Many Requests: Rate limit exceeded for {path}. Retry after {reset_secs} seconds.",
            headers={
                "Retry-After": str(reset_secs),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_secs)
            }
        )

    response = await call_next(request)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    return response
