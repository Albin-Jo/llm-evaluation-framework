from typing import Dict

from fastapi import APIRouter, Depends, status, Request
from pydantic import BaseModel

from backend.app.api.dependencies.auth import get_required_current_user
from backend.app.api.middleware.jwt_validator import UserContext

auth_router = APIRouter()


class UserResponse(BaseModel):
    """Schema for user response."""
    sub: str
    username: str
    email: str
    name: str
    roles: list[str]


class AuthStatusResponse(BaseModel):
    """Schema for authentication status response."""
    authenticated: bool
    user: Dict = None


@auth_router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: UserContext = Depends(get_required_current_user)):
    """
    Get information about the currently authenticated user.

    Returns:
        UserResponse: Current user information
    """
    return UserResponse(
        sub=current_user.sub,
        username=current_user.preferred_username,
        email=current_user.email,
        name=current_user.name,
        roles=current_user.roles
    )


@auth_router.get("/status", response_model=AuthStatusResponse)
async def auth_status(request: Request):
    """
    Check if the user is authenticated.

    Returns:
        AuthStatusResponse: Authentication status and user info if authenticated
    """
    user = getattr(request.state, "user", None)

    if user:
        return AuthStatusResponse(
            authenticated=True,
            user={
                "sub": user.sub,
                "username": user.preferred_username,
                "email": user.email,
                "name": user.name,
                "roles": user.roles
            }
        )

    return AuthStatusResponse(authenticated=False)


@auth_router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout():
    """
    Logout the current user.

    This endpoint doesn't actually invalidate the token (that happens on the client side)
    but is provided as a convenient endpoint for clients.

    Returns:
        204 No Content
    """
    # In a stateful auth system, we would invalidate the session here
    # Since we're using JWT tokens, there's nothing to do on the server side
    return None
