# File: backend/app/api/middleware/jwt_validator.py

import logging
import time
import base64
from typing import Dict, Optional, List
from fastapi import Request, HTTPException, status
from jose import jwt, jwk, JWTError, ExpiredSignatureError
from jose.utils import base64url_decode
import httpx
from jose.backends.cryptography_backend import CryptographyRSAKey

from backend.app.core.config import settings
from backend.app.db.models.orm import User
from backend.app.db.repositories.user_repository import UserRepository
from backend.app.db.session import db_session

logger = logging.getLogger(__name__)

# Cache for JWKS
_jwks_cache = {
    "keys": None,
    "last_updated": 0,
    "expires_in": 3600  # Cache expiration in seconds
}


async def get_jwks():
    """
    Fetch and cache JSON Web Key Set (JWKS) from Keycloak.
    """
    current_time = time.time()

    # Return cached JWKS if still valid
    if (_jwks_cache["keys"] is not None and
            current_time - _jwks_cache["last_updated"] < _jwks_cache["expires_in"]):
        return _jwks_cache["keys"]

    # Otherwise fetch new JWKS
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(settings.KEYCLOAK_AUTHORITY)
            response.raise_for_status()

            jwks_data = response.json()
            logger.info(f"jwks: {jwks_data}")

            # Create a synthetic JWKS from the public key if we get a realm config instead
            # of a standard JWKS response
            if "public_key" in jwks_data and "keys" not in jwks_data:
                logger.info("Creating synthetic JWKS from public key in realm config")

                # Get the public key from realm config
                public_key = jwks_data.get("public_key", "")

                # Store as a simplified JWK
                synthetic_key = {
                    "kid": "default-key",
                    "kty": "RSA",
                    "alg": "RS256",
                    "use": "sig",
                    "n": public_key,  # Store original key for flexibility
                    "e": "AQAB"
                }

                synthetic_keys = [synthetic_key]
                _jwks_cache["keys"] = synthetic_keys
                _jwks_cache["last_updated"] = current_time

                return _jwks_cache["keys"]

            # Standard JWKS handling if we have a "keys" array
            if "keys" in jwks_data:
                keys = jwks_data.get("keys", [])
                _jwks_cache["keys"] = keys
                _jwks_cache["last_updated"] = current_time
                return _jwks_cache["keys"]

            # If we got here, we don't have keys or a public key
            logger.error("No keys or public key found in response")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Invalid authentication key format"
            )

    except Exception as e:
        logger.error(f"Error fetching JWKS: {str(e)}")
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
            db_user: Optional[User] = None
    ):
        self.sub = sub
        self.preferred_username = preferred_username
        self.email = email
        self.name = name
        self.roles = roles or []
        self.db_user = db_user

    @classmethod
    def from_token_payload(cls, payload: Dict):
        """
        Create UserContext from token payload.
        """
        # Extract roles from token
        roles = []

        # Extract from realm_access if available
        realm_access = payload.get("realm_access", {})
        if realm_access and "roles" in realm_access:
            roles.extend(realm_access.get("roles", []))

        # Extract from resource_access if available
        resource_access = payload.get("resource_access", {})
        if resource_access:
            for resource, resource_data in resource_access.items():
                if resource_data and "roles" in resource_data:
                    roles.extend(resource_data.get("roles", []))

        return cls(
            sub=payload.get("sub", ""),
            preferred_username=payload.get("preferred_username", ""),
            email=payload.get("email", ""),
            name=payload.get("name", ""),
            roles=roles
        )


async def verify_token(token: str) -> Dict:
    """
    Verify JWT token and return payload.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Get token header to extract kid (key ID)
    try:
        headers = jwt.get_unverified_header(token)
        logger.info(f"headers: {headers}")
        kid = headers.get("kid")
        logger.info(f"kid: {kid}")

        # Fix: Log comparison to debug potential kid issues
        if kid and kid.startswith("zA37w"):
            logger.warning(f"Kid value may be corrupted: {kid}")
            # Try to fix known prefix issue
            kid = kid.replace("zA37w", "")
            logger.info(f"Corrected kid: {kid}")
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token header",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Get JSON Web Keys
    jwks = await get_jwks()

    # Check if we got a valid JWKS
    if not isinstance(jwks, list):
        logger.error(f"Unexpected JWKS format: {type(jwks)}")
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
        logger.info("Using synthetic default key regardless of kid in token")
        key = jwks[0]
    else:
        # Standard kid matching
        for jwk_key in jwks:
            if jwk_key and jwk_key.get("kid") == kid:  # Check if jwk_key is not None
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

        # For synthetic JWKS with key in 'n'
        if key.get("kid") == "default-key":
            logger.info("Using synthetic key for token validation")
            public_key = key.get("n", "")

            # Try different approaches to use the public key
            try:
                # Approach 1: Try to use it as a JWKS directly
                payload = jwt.decode(
                    token,
                    {"keys": [key]},
                    algorithms=["RS256"]
                )
                logger.info("Successfully validated with JWKS approach")

            except Exception as e1:
                logger.warning(f"JWKS approach failed: {str(e1)}")

                try:
                    # Approach 2: Try to format as PEM if not already
                    pem_key = public_key
                    if not pem_key.startswith("-----BEGIN"):
                        # If the key is just the base64 content without headers
                        pem_key = f"-----BEGIN PUBLIC KEY-----\n{public_key}\n-----END PUBLIC KEY-----"

                    payload = jwt.decode(
                        token,
                        pem_key,
                        algorithms=["RS256"]
                    )
                    logger.info("Successfully validated with PEM approach")

                except Exception as e2:
                    logger.warning(f"PEM approach failed: {str(e2)}")

                    try:
                        # Approach 3: Try using jose.jwk.construct
                        rsa_key = {
                            "kty": "RSA",
                            "n": key.get("n"),
                            "e": key.get("e", "AQAB")
                        }
                        constructed_key = CryptographyRSAKey(rsa_key, "RS256")

                        payload = jwt.decode(
                            token,
                            constructed_key,
                            algorithms=["RS256"]
                        )
                        logger.info("Successfully validated with constructed key approach")

                    except Exception as e3:
                        logger.error(f"Constructed key approach failed: {str(e3)}")

                        # Fallback to unverified payload for development only
                        # WARNING: Remove this in production!
                        logger.warning("USING UNVERIFIED PAYLOAD - SECURITY RISK!")
                        payload = jwt.get_unverified_claims(token)
        else:
            # Standard JWK usage
            logger.info("Using standard JWK for token validation")
            payload = jwt.decode(
                token,
                key,
                algorithms=["RS256"]
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


async def jwt_auth_middleware(request: Request, call_next):
    """
    Middleware to validate JWT tokens and add user context to request state.
    """
    # Skip authentication for certain paths
    skip_auth_paths = [
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/auth/login",
        "/api/auth/refresh",
    ]

    for path in skip_auth_paths:
        if request.url.path.startswith(path):
            return await call_next(request)

    # Extract token from Authorization header
    authorization = request.headers.get("Authorization")

    if not authorization:
        # Skip auth for now if no token provided (will implement strict mode later)
        return await call_next(request)

    token_type, _, token = authorization.partition(" ")

    if token_type.lower() != "bearer":
        # Skip auth for non-Bearer tokens
        return await call_next(request)

    try:
        # Verify token
        payload = await verify_token(token)

        # Create user context
        user_context = UserContext.from_token_payload(payload)

        # Sync user with database (in background)
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

        # Return the actual HTTP exception
        return HTTPException(
            status_code=e.status_code,
            detail=e.detail,
            headers=e.headers
        )(request)

    except Exception as e:
        logger.error(f"Unexpected error in JWT middleware: {str(e)}")
        # Return a generic error
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during authentication"
        )(request)