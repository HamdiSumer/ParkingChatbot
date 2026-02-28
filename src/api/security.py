"""API Security module for the parking admin dashboard.

Provides:
- API key authentication
- Request validation
- Rate limiting (basic)
"""
import secrets
import time
from typing import Optional
from collections import defaultdict
from fastapi import HTTPException, Security, Request
from fastapi.security import APIKeyHeader, APIKeyQuery
from starlette.status import HTTP_403_FORBIDDEN, HTTP_429_TOO_MANY_REQUESTS

from src.config import get_config
from src.utils.logging import logger

config = get_config()

# API key can be passed via header or query parameter
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
api_key_query = APIKeyQuery(name="api_key", auto_error=False)


# Simple in-memory rate limiter
class RateLimiter:
    """Simple rate limiter to prevent abuse."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        """Check if a request from this IP is allowed."""
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if t > window_start
        ]

        # Check if under limit
        if len(self.requests[client_ip]) >= self.max_requests:
            return False

        # Record this request
        self.requests[client_ip].append(now)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)


def get_client_ip(request: Request) -> str:
    """Extract client IP from request."""
    # Check for forwarded header (behind proxy)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def verify_api_key(
    request: Request,
    api_key_header: Optional[str] = Security(api_key_header),
    api_key_query: Optional[str] = Security(api_key_query),
) -> bool:
    """Verify API key from header or query parameter.

    Args:
        request: FastAPI request object
        api_key_header: API key from X-API-Key header
        api_key_query: API key from query parameter

    Returns:
        True if authentication succeeds

    Raises:
        HTTPException: If authentication fails
    """
    # Check rate limiting first
    client_ip = get_client_ip(request)
    if not rate_limiter.is_allowed(client_ip):
        logger.warning(f"Rate limit exceeded for {client_ip}")
        raise HTTPException(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please try again later.",
        )

    # If API key requirement is disabled, allow all requests
    if not config.REQUIRE_API_KEY:
        return True

    # Get API key from either source
    api_key = api_key_header or api_key_query

    if not api_key:
        logger.warning(f"Missing API key from {client_ip}")
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="API key required. Provide via X-API-Key header or api_key query parameter.",
        )

    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(api_key, config.ADMIN_API_KEY):
        logger.warning(f"Invalid API key from {client_ip}")
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    logger.debug(f"API key verified for {client_ip}")
    return True


async def verify_api_key_optional(
    request: Request,
    api_key_header: Optional[str] = Security(api_key_header),
    api_key_query: Optional[str] = Security(api_key_query),
) -> bool:
    """Optional API key verification (logs but doesn't block).

    Use this for endpoints that should work without auth but benefit from it.
    """
    try:
        return await verify_api_key(request, api_key_header, api_key_query)
    except HTTPException:
        return False
