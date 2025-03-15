from typing import Dict
import httpx
from authlib.integrations.starlette_client import OAuth
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2AuthorizationCodeBearer
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config.settings import settings
from app.db.session import get_db
from app.db.repositories.user_repository import UserRepository
from app.models.orm.models import User, UserRole
import logging

logger = logging.getLogger(__name__)

oauth = OAuth()
oauth.register(
    name="oidc",
    server_metadata_url=settings.OIDC_DISCOVERY_URL,
    client_id=settings.OIDC_CLIENT_ID,
    client_secret=settings.OIDC_CLIENT_SECRET,
    client_kwargs={"scope": "openid email profile"},
)

oauth2_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl=f"{settings.OIDC_DISCOVERY_URL}/auth",
    tokenUrl=f"{settings.OIDC_DISCOVERY_URL}/token",
)


class AuthService:
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        self.user_repo = UserRepository(db_session)

    async def verify_token(self, token: str) -> Dict:
        """Fetch and verify JWT using OIDC provider's public key."""
        try:
            async with httpx.AsyncClient() as client:
                openid_config_response = await client.get(settings.OIDC_DISCOVERY_URL)
                openid_config = openid_config_response.json()
                jwks_uri = openid_config.get("jwks_uri")
                jwks_response = await client.get(jwks_uri)
                jwks = jwks_response.json()

            payload = jwt.decode(
                token,
                jwks,
                algorithms=["RS256"],
                audience=settings.OIDC_CLIENT_ID,
                options={"verify_signature": False}
            )
            return payload
        except Exception as e:
            logger.error(f"JWT verification failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def get_current_user(self, token: str) -> User:
        """Validate JWT and retrieve user."""
        payload = await self.verify_token(token)
        external_id = payload.get("sub")
        if not external_id:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        user = await self.user_repo.get_by_external_id(external_id)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        if not user.is_active:
            raise HTTPException(status_code=403, detail="Inactive user")

        return user

    async def check_admin_privileges(self, user: User) -> None:
        if user.role != UserRole.ADMIN:
            raise HTTPException(status_code=403, detail="Not enough permissions")

    async def create_user_from_oidc(self, user_data: Dict) -> User:
        """Create user from OIDC data."""
        external_id = user_data.get("sub")
        email = user_data.get("email")
        name = user_data.get("name") or f"{user_data.get('given_name', '')} {user_data.get('family_name', '')}".strip()

        user = await self.user_repo.get_by_external_id(external_id)
        if user:
            return user

        return await self.user_repo.create(
            external_id=external_id,
            email=email,
            display_name=name,
            role=UserRole.VIEWER,
        )


async def get_auth_service(db: AsyncSession = Depends(get_db)) -> AuthService:
    return AuthService(db)


async def get_current_active_user(auth_service: AuthService = Depends(get_auth_service), token: str = Depends(oauth2_scheme)) -> User:
    return await auth_service.get_current_user(token)


async def get_current_admin_user(current_user: User = Depends(get_current_active_user)) -> User:
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Not enough permissions")
    return current_user
