import logging
import time
from functools import wraps
from typing import Dict, Optional, List, Any

import httpx
from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from jose import jwt, JWTError, ExpiredSignatureError

from backend.app.core.config import settings
from backend.app.db.models.orm import User
from backend.app.db.repositories.user_repository import UserRepository
from backend.app.db.session import db_session

logger = logging.getLogger(__name__)


def timed_function(func):
    """
    Decorator to time function execution and log.json the results.
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.debug(f"{func.__name__} executed in {execution_time:.2f}s")
        return result

    return wrapper


# Cache for JWKS
_jwks_cache = {
    "keys": None,
    "last_updated": 0,
    "expires_in": getattr(settings, "JWKS_CACHE_TTL", 3600),  # Configurable cache TTL
    "stats": {
        "hits": 0,
        "misses": 0,
        "fetch_times": [],  # Only keep last 10 fetch times
        "errors": 0
    },
    "circuit_breaker": {
        "failures": 0,
        "max_failures": 3,
        "open_until": 0,
        "backoff_multiplier": 1.5,
        "base_backoff": 30  # seconds
    }
}


def update_cache_stats(hit=False, fetch_time=None, error=False):
    """
    Update JWKS cache statistics with bounded counters.

    Args:
        hit: Whether this was a cache hit
        fetch_time: Time taken to fetch JWKS, if applicable
        error: Whether an error occurred
    """
    if hit:
        _jwks_cache["stats"]["hits"] = (_jwks_cache["stats"]["hits"] + 1) % 1_000_000  # Prevent overflow
    else:
        _jwks_cache["stats"]["misses"] = (_jwks_cache["stats"]["misses"] + 1) % 1_000_000

    if fetch_time is not None:
        # Keep only last 10 fetch times for average calculation
        _jwks_cache["stats"]["fetch_times"].append(fetch_time)
        if len(_jwks_cache["stats"]["fetch_times"]) > 10:
            _jwks_cache["stats"]["fetch_times"].pop(0)

    if error:
        _jwks_cache["stats"]["errors"] = (_jwks_cache["stats"]["errors"] + 1) % 1_000_000


@timed_function
async def get_jwks() -> List[Dict[str, Any]]:
    """
    Fetch and cache JSON Web Key Set (JWKS) from Keycloak.

    Returns:
        List[Dict[str, Any]]: A list of JWK objects

    Raises:
        HTTPException: If JWKS cannot be fetched or is invalid
    """
    current_time = time.time()

    # Check circuit breaker
    if _jwks_cache["circuit_breaker"]["open_until"] > current_time:
        logger.warning(
            f"Circuit breaker open until {_jwks_cache['circuit_breaker']['open_until'] - current_time:.1f}s from now, using cached keys")
        if _jwks_cache["keys"] is not None:
            return _jwks_cache["keys"]
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable"
        )

    # Return cached JWKS if still valid
    if (_jwks_cache["keys"] is not None and
            current_time - _jwks_cache["last_updated"] < _jwks_cache["expires_in"]):
        cache_age = int(current_time - _jwks_cache["last_updated"])
        update_cache_stats(hit=True)

        # Log every 100 hits or at debug level
        if _jwks_cache["stats"]["hits"] % 100 == 0:
            logger.info(
                f"JWKS cache stats: {_jwks_cache['stats']['hits']} hits, {_jwks_cache['stats']['misses']} misses, {_jwks_cache['stats']['errors']} errors")

        logger.debug(
            f"JWKS cache hit (age: {cache_age}s, "
            f"expires in: {_jwks_cache['expires_in'] - cache_age}s)")
        return _jwks_cache["keys"]

    update_cache_stats(hit=False)
    logger.info(f"JWKS cache miss, fetching fresh JWKS from authority")

    # Otherwise fetch new JWKS
    try:
        start_time = time.time()
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(settings.KEYCLOAK_AUTHORITY)
            response.raise_for_status()

            fetch_time = time.time() - start_time
            update_cache_stats(fetch_time=fetch_time)

            avg_fetch_time = sum(_jwks_cache["stats"]["fetch_times"]) / len(_jwks_cache["stats"]["fetch_times"]) if \
                _jwks_cache["stats"]["fetch_times"] else 0

            jwks_data = response.json()
            logger.info(
                f"Retrieved JWKS data from authority in {fetch_time:.2f}s (avg: {avg_fetch_time:.2f}s)")

            # Handle standard JWKS format with "keys" array
            if "keys" in jwks_data:
                keys = jwks_data.get("keys", [])
                _jwks_cache["keys"] = keys
                _jwks_cache["last_updated"] = current_time
                # Reset circuit breaker on successful fetch
                _jwks_cache["circuit_breaker"]["failures"] = 0
                logger.info(f"Cached standard JWKS with {len(keys)} keys, next refresh in {_jwks_cache['expires_in']}s")
                return _jwks_cache["keys"]

            # Handle Keycloak realm config format with public_key
            elif "public_key" in jwks_data:
                public_key = jwks_data.get("public_key", "")

                # Create a JWK from the public key
                synthetic_key = {
                    "kid": "default-key",
                    "kty": "RSA",
                    "alg": "RS256",
                    "use": "sig",
                    "n": public_key,
                    "e": "AQAB"
                }

                _jwks_cache["keys"] = [synthetic_key]
                _jwks_cache["last_updated"] = current_time
                # Reset circuit breaker on successful fetch
                _jwks_cache["circuit_breaker"]["failures"] = 0
                logger.info(f"Cached synthetic JWKS from public key, next refresh in {_jwks_cache['expires_in']}s")
                return _jwks_cache["keys"]

            # No valid keys found
            else:
                logger.error("No keys or public key found in JWKS response")
                update_cache_stats(error=True)
                # Update circuit breaker
                _jwks_cache["circuit_breaker"]["failures"] += 1
                if _jwks_cache["circuit_breaker"]["failures"] >= _jwks_cache["circuit_breaker"]["max_failures"]:
                    backoff = _jwks_cache["circuit_breaker"]["base_backoff"] * (
                            _jwks_cache["circuit_breaker"]["backoff_multiplier"] **
                            (_jwks_cache["circuit_breaker"]["failures"] - _jwks_cache["circuit_breaker"][
                                "max_failures"])
                    )
                    _jwks_cache["circuit_breaker"]["open_until"] = current_time + min(backoff, 300)  # Max 5 minutes

                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Invalid authentication key format"
                )

    except Exception as e:
        logger.error(f"Error fetching JWKS: {str(e)}")
        update_cache_stats(error=True)

        # Update circuit breaker
        _jwks_cache["circuit_breaker"]["failures"] += 1
        if _jwks_cache["circuit_breaker"]["failures"] >= _jwks_cache["circuit_breaker"]["max_failures"]:
            backoff = _jwks_cache["circuit_breaker"]["base_backoff"] * (
                    _jwks_cache["circuit_breaker"]["backoff_multiplier"] **
                    (_jwks_cache["circuit_breaker"]["failures"] - _jwks_cache["circuit_breaker"]["max_failures"])
            )
            _jwks_cache["circuit_breaker"]["open_until"] = current_time + min(backoff, 300)  # Max 5 minutes
            logger.warning(
                f"Circuit breaker opened for {min(backoff, 300):.1f}s after {_jwks_cache['circuit_breaker']['failures']} failures")

        # Return cached keys if available, otherwise raise error
        if _jwks_cache["keys"] is not None:
            logger.warning("Using cached JWKS due to fetch error")
            return _jwks_cache["keys"]
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to fetch authentication keys"
        )


class UserContext:
    """
    User context extracted from JWT token.
    """

    def __init__(
            self,
            sub: str,
            preferred_username: str,
            email: Optional[str] = None,
            name: Optional[str] = None,
            roles: Optional[List[str]] = None,
            db_user: Optional[User] = None,
            token: Optional[str] = None
    ):
        self.sub = sub
        self.preferred_username = preferred_username
        self.email = email
        self.name = name
        self.roles = roles or []
        self.db_user = db_user
        self.token = token  # Store the original token for use with external services

    @classmethod
    def from_token_payload(cls, payload: Dict[str, Any], token: Optional[str] = None) -> "UserContext":
        """
        Create UserContext from token payload with flexible role extraction.

        Args:
            payload: The decoded JWT payload
            token: Original JWT token string

        Returns:
            UserContext: User context object with token data
        """
        # Extract roles from token with safer access patterns
        roles = []

        # Extract from realm_access if available
        realm_access = payload.get("realm_access", {})
        if isinstance(realm_access, dict) and isinstance(realm_access.get("roles", []), list):
            roles.extend(realm_access.get("roles", []))

        # Extract from resource_access if available
        resource_access = payload.get("resource_access", {})
        if isinstance(resource_access, dict):
            for resource, resource_data in resource_access.items():
                if isinstance(resource_data, dict) and isinstance(resource_data.get("roles", []), list):
                    roles.extend(resource_data.get("roles", []))

        # Support multiple role formats
        # Some systems use a simple 'roles' array at root level
        if isinstance(payload.get("roles", []), list):
            roles.extend(payload.get("roles", []))

        # Alternative role format
        if isinstance(payload.get("authorities", []), list):
            roles.extend(payload.get("authorities", []))

        return cls(
            sub=payload.get("sub", ""),
            preferred_username=payload.get("preferred_username", "") or payload.get("username", ""),
            email=payload.get("email", ""),
            name=payload.get("name", "") or payload.get("display_name", ""),
            roles=roles,
            token=token  # Include the original token
        )


def format_public_key_as_pem(public_key: str) -> str:
    """
    Format a public key as PEM if it's not already.

    Args:
        public_key: The public key string

    Returns:
        str: Properly formatted PEM public key
    """
    if not public_key.startswith("-----BEGIN"):
        # If the key is just the base64 content without headers
        return f"-----BEGIN PUBLIC KEY-----\n{public_key}\n-----END PUBLIC KEY-----"
    return public_key


@timed_function
async def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify JWT token and return payload.

    Args:
        token: JWT token string

    Returns:
        Dict[str, Any]: Decoded token payload

    Raises:
        HTTPException: If token is invalid or verification fails
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Log token verification attempt with partial token for debugging
    token_prefix = token[:10] if len(token) > 10 else token
    logger.debug(f"Verifying token: {token_prefix}...")

    # Get token header to extract kid (key ID)
    try:
        headers = jwt.get_unverified_header(token)
        kid = headers.get("kid")
        algorithm = headers.get("alg", "RS256")

        if algorithm not in ["RS256", "RS384", "RS512"]:
            logger.error(f"Unsupported token algorithm: {algorithm}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unsupported token algorithm",
                headers={"WWW-Authenticate": "Bearer"}
            )

    except JWTError:
        logger.error("Failed to parse token header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token header",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Get JSON Web Keys
    jwks = await get_jwks()

    # Check if we got a valid JWKS
    if not isinstance(jwks, list):
        logger.error(f"Invalid JWKS format: {type(jwks)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid JWKS format",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # If we have no keys, we can't validate
    if not jwks:
        logger.error("No keys available for token validation")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No authentication keys available",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Find the key with matching kid
    key = None

    # If we're using a synthetic JWKS with a default key and the token has a kid,
    # we might need to ignore the kid mismatch
    if len(jwks) == 1 and jwks[0].get("kid") == "default-key":
        logger.debug("Using synthetic default key regardless of kid in token")
        key = jwks[0]
    else:
        # Standard kid matching
        for jwk_key in jwks:
            if jwk_key and jwk_key.get("kid") == kid:
                key = jwk_key
                break

    if not key:
        logger.error(f"No matching key found for kid: {kid}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token signing key not found",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Verify the token
    try:
        # Check for key properties
        if not key.get("kty"):
            logger.error("Key type not found in JWK")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid key format",
                headers={"WWW-Authenticate": "Bearer"}
            )

        # Determine if we're using a synthetic key
        is_synthetic_key = key.get("kid") == "default-key"

        try:
            # For synthetic keys with custom formatting
            if is_synthetic_key:
                public_key = key.get("n", "")
                pem_key = format_public_key_as_pem(public_key)

                payload = jwt.decode(
                    token,
                    pem_key,
                    algorithms=[algorithm],
                    audience=getattr(settings, "TOKEN_AUDIENCE", None),
                    issuer=getattr(settings, "TOKEN_ISSUER", None),
                    options={
                        "verify_signature": True,
                        "verify_aud": hasattr(settings, "TOKEN_AUDIENCE"),
                        "verify_iss": hasattr(settings, "TOKEN_ISSUER"),
                        "verify_exp": True,
                        "verify_nbf": True,
                        "verify_iat": True,
                        "require_exp": True,
                    }
                )
            else:
                # Standard JWK usage
                payload = jwt.decode(
                    token,
                    key,
                    algorithms=[algorithm],
                    audience=getattr(settings, "TOKEN_AUDIENCE", None),
                    issuer=getattr(settings, "TOKEN_ISSUER", None),
                    options={
                        "verify_signature": True,
                        "verify_aud": hasattr(settings, "TOKEN_AUDIENCE"),
                        "verify_iss": hasattr(settings, "TOKEN_ISSUER"),
                        "verify_exp": True,
                        "verify_nbf": True,
                        "verify_iat": True,
                        "require_exp": True,
                    }
                )
        except Exception as decode_error:
            logger.error(f"Token decode error: {str(decode_error)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token signature",
                headers={"WWW-Authenticate": "Bearer"}
            )

        logger.info(f"Token validation successful for subject: {payload.get('sub', 'unknown')}")
        return payload

    except ExpiredSignatureError:
        logger.warning("Token has expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except JWTError as e:
        logger.error(f"JWT validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def create_error_response(status_code: int, detail: str,
                                headers: Optional[Dict[str, str]] = None) -> JSONResponse:
    """
    Create a proper FastAPI response for errors in middleware.

    Args:
        status_code: HTTP status code
        detail: Error detail message
        headers: Optional response headers

    Returns:
        JSONResponse: The formatted error response
    """
    content = {"detail": detail}
    return JSONResponse(
        status_code=status_code,
        content=content,
        headers=headers or {}
    )


async def jwt_auth_middleware(request: Request, call_next):
    """
    Middleware to validate JWT tokens and add user context to request state.

    Args:
        request: FastAPI request object
        call_next: Next middleware in the chain

    Returns:
        The response from the next middleware
    """
    # Skip authentication for certain paths
    skip_auth_paths = [
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/auth/login",
        "/api/auth/refresh",
    ]

    # Check if path is in the skip list
    for path in skip_auth_paths:
        if request.url.path.startswith(path):
            return await call_next(request)

    # Extract token from Authorization header
    authorization = request.headers.get("Authorization")

    # Check if strict auth mode is enabled
    strict_auth_mode = getattr(settings, "STRICT_AUTH_MODE", False)

    if not authorization:
        if strict_auth_mode:
            # Create a proper response for unauthorized request
            return await create_error_response(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing authentication token",
                headers={"WWW-Authenticate": "Bearer"}
            )
        else:
            # Allow request without token if not in strict mode
            return await call_next(request)

    # Parse token from header
    token_parts = authorization.split()
    if len(token_parts) != 2 or token_parts[0].lower() != "bearer":
        return await create_error_response(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Expected 'Bearer {token}'",
            headers={"WWW-Authenticate": "Bearer"}
        )

    token = token_parts[1]

    # Store raw token in request state for use by endpoints
    request.state.jwt_token = token

    try:
        # Verify token
        payload = await verify_token(token)

        # Create user context, passing the original token
        user_context = UserContext.from_token_payload(payload, token)

        # Sync user with database
        async with db_session() as session:
            user_repo = UserRepository(session)
            user = await user_repo.get_by_external_id(user_context.sub)

            if not user:
                # Create user if not exists
                user = await user_repo.create(
                    external_id=user_context.sub,
                    email=user_context.email or f"{user_context.preferred_username}@example.com",
                    display_name=user_context.name or user_context.preferred_username
                )

            # Update user context with database user
            user_context.db_user = user

            # Store user context in request state
            request.state.user = user_context

        return await call_next(request)

    except HTTPException as e:
        # Log the error
        logger.warning(f"Authentication error: {e.detail}")
        # Return a properly formatted response instead of calling the exception
        return await create_error_response(
            status_code=e.status_code,
            detail=e.detail,
            headers=e.headers
        )

    except Exception as e:
        logger.error(f"Unexpected error in JWT middleware: {str(e)}")
        return await create_error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during authentication"
        )


async def check_token_expiration(token: str) -> Dict[str, Any]:
    """
    Check if a token is nearing expiration and should be refreshed.

    Args:
        token: JWT token string

    Returns:
        Dict with expiration info: 
        {
            "expires_at": timestamp,
            "expires_in_seconds": seconds,
            "should_refresh": boolean
        }
    """
    try:
        # Get payload without verifying signature - just to check exp
        payload = jwt.get_unverified_claims(token)

        if "exp" not in payload:
            return {
                "expires_at": None,
                "expires_in_seconds": None,
                "should_refresh": True  # No expiration found, safer to refresh
            }

        exp_timestamp = payload["exp"]
        current_time = time.time()
        time_remaining = exp_timestamp - current_time

        # Configure the threshold for refresh (e.g., 5 minutes before expiration)
        refresh_threshold = getattr(settings, "TOKEN_REFRESH_THRESHOLD_SECONDS", 300)

        return {
            "expires_at": exp_timestamp,
            "expires_in_seconds": time_remaining,
            "should_refresh": time_remaining < refresh_threshold
        }

    except Exception as e:
        logger.warning(f"Error checking token expiration: {str(e)}")
        return {
            "expires_at": None,
            "expires_in_seconds": None,
            "should_refresh": True  # Error checking, safer to refresh
        }
