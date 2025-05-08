import time
from typing import Dict, Optional, Callable, Any

from fastapi import HTTPException, Request, status


# Simple in-memory store for rate limiting
# In production, use Redis or another distributed cache
class RateLimitStore:
    def __init__(self):
        self.store: Dict[str, Dict[str, Any]] = {}

    def get_key(self, key: str) -> Optional[Dict[str, Any]]:
        """Get rate limit data for a key"""
        self._cleanup()
        return self.store.get(key)

    def increment(self, key: str, window_seconds: int) -> Dict[str, Any]:
        """Increment rate limit counter for a key"""
        self._cleanup()
        now = time.time()

        if key not in self.store:
            self.store[key] = {
                "count": 0,
                "window_start": now,
                "window_end": now + window_seconds
            }

        data = self.store[key]

        # If window expired, reset counter
        if now > data["window_end"]:
            data["count"] = 0
            data["window_start"] = now
            data["window_end"] = now + window_seconds

        # Increment counter
        data["count"] += 1
        return data

    def _cleanup(self):
        """Remove expired entries"""
        now = time.time()
        expired_keys = [k for k, v in self.store.items() if v["window_end"] < now]
        for k in expired_keys:
            del self.store[k]


# Global store instance
rate_limit_store = RateLimitStore()


def rate_limit(max_requests: int = 100, period_seconds: int = 60, by_ip: bool = True) -> Callable:
    """
    FastAPI dependency for rate limiting.

    Args:
        max_requests: Maximum number of requests allowed in the time period
        period_seconds: Time period in seconds
        by_ip: Whether to limit by IP address

    Returns:
        Callable: FastAPI dependency function
    """

    async def rate_limit_dependency(request: Request):
        # Determine the client identifier
        if by_ip:
            client_id = request.client.host if request.client else "unknown"
        else:
            # Could use auth token, user ID, etc.
            client_id = "global"

        # Create a unique key for this endpoint and client
        endpoint = f"{request.method}:{request.url.path}"
        key = f"{client_id}:{endpoint}"

        # Update rate limit counter
        data = rate_limit_store.increment(key, period_seconds)

        # Check if limit exceeded
        if data["count"] > max_requests:
            # Calculate time until reset
            reset_after = int(data["window_end"] - time.time())

            # Set rate limit headers
            headers = {
                "X-RateLimit-Limit": str(max_requests),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_after),
                "Retry-After": str(reset_after)
            }

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {reset_after} seconds.",
                headers=headers
            )

        # Set rate limit headers for successful requests too
        request.state.rate_limit_headers = {
            "X-RateLimit-Limit": str(max_requests),
            "X-RateLimit-Remaining": str(max_requests - data["count"]),
            "X-RateLimit-Reset": str(int(data["window_end"] - time.time()))
        }

    return rate_limit_dependency


# Middleware to add rate limit headers to responses
async def rate_limit_middleware(request: Request, call_next):
    response = await call_next(request)

    # Add rate limit headers if available
    if hasattr(request.state, "rate_limit_headers"):
        for key, value in request.state.rate_limit_headers.items():
            response.headers[key] = value

    return response
