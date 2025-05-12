from typing import Optional, Tuple

from fastapi import Request, HTTPException, status

from backend.app.api.middleware.jwt_validator import UserContext


async def get_current_user(request: Request) -> Optional[UserContext]:
    """
    Get current user from request state.

    This dependency extracts the authenticated user from the request state.
    If no user is authenticated, it returns None.

    Args:
        request: The FastAPI request object

    Returns:
        Optional[UserContext]: The current user context or None
    """
    return getattr(request.state, "user", None)


async def get_optional_current_user(request: Request) -> Optional[UserContext]:
    """
    Get current user from request state, returning None if not authenticated.

    This dependency is useful for endpoints that can work with or without authentication.

    Args:
        request: The FastAPI request object

    Returns:
        Optional[UserContext]: The current user context or None
    """
    return getattr(request.state, "user", None)


async def get_required_current_user(request: Request) -> UserContext:
    """
    Get current user from request state, raising 401 if not authenticated.

    This dependency requires the user to be authenticated.

    Args:
        request: The FastAPI request object

    Returns:
        UserContext: The current user context

    Raises:
        HTTPException: If the user is not authenticated
    """
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # If the UserContext doesn't have the token but it's in request state, add it
    if not hasattr(user, "token") or not user.token:
        jwt_token = getattr(request.state, "jwt_token", None)
        if jwt_token and isinstance(user, UserContext):
            user.token = jwt_token

    return user


async def get_jwt_token(request: Request) -> Optional[str]:
    """
    Get JWT token from request state.

    Args:
        request: The FastAPI request object

    Returns:
        Optional[str]: JWT token or None if not present
    """
    return getattr(request.state, "jwt_token", None)


async def get_user_and_token(request: Request) -> Tuple[Optional[UserContext], Optional[str]]:
    """
    Get both user context and JWT token.

    Args:
        request: The FastAPI request object

    Returns:
        Tuple[Optional[UserContext], Optional[str]]: User context and JWT token
    """
    user = getattr(request.state, "user", None)
    token = getattr(request.state, "jwt_token", None)
    return user, token
