# File: app/api/v1/auth.py

from fastapi import APIRouter, Depends, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.db.session import get_db
from backend.app.db.models.orm import User
from backend.app.db.schema.user_schema import UserResponse
from backend.app.services.auth import AuthService, get_auth_service, get_current_active_user, oauth

router = APIRouter()


@router.get("/login")
async def login(request: Request):
    """
    Initiate OIDC login flow.

    This endpoint redirects the user to the OIDC provider's login page.
    """
    redirect_uri = f"{settings.APP_BASE_URL}/api/auth/callback"
    return await oauth.oidc.authorize_redirect(request, redirect_uri)


@router.get("/callback")
async def auth_callback(
        request: Request,
        db: AsyncSession = Depends(get_db),
        auth_service: AuthService = Depends(get_auth_service)
):
    """
    Handle OIDC callback after successful authentication.

    This endpoint exchanges the authorization code for tokens and creates/updates the user.
    """
    # Get token from OIDC provider
    token = await oauth.oidc.authorize_access_token(request)
    user_info = await oauth.oidc.parse_id_token(request, token)

    # Create or update user in our database
    user = await auth_service.create_user_from_oidc(user_info)

    # Create session or JWT for our app
    # In a real app, you might create a session or JWT here

    # Redirect to frontend_v1 with token
    frontend_url = f"{settings.APP_BASE_URL}?token={token['access_token']}"
    return RedirectResponse(url=frontend_url)


@router.get("/me", response_model=UserResponse)
async def get_user_me(current_user: User = Depends(get_current_active_user)):
    """
    Get current user information.
    """
    return current_user


@router.get("/logout")
async def logout():
    """
    Log out the current user.

    This endpoint redirects the user to the OIDC provider's logout page.
    """
    # In a real app, you might invalidate the session or JWT here

    # Redirect to OIDC provider's logout endpoint
    logout_url = f"{settings.OIDC_DISCOVERY_URL}/logout"
    post_logout_redirect_uri = settings.APP_BASE_URL
    return RedirectResponse(
        url=f"{logout_url}?client_id={settings.OIDC_CLIENT_ID}&post_logout_redirect_uri={post_logout_redirect_uri}"
    )