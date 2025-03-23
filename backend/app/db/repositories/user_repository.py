# File: app/db/repositories/user_repository.py
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.orm.models import User, UserRole


class UserRepository:
    """Repository for User model operations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID."""
        query = select(User).where(User.id == user_id)
        result = await self.session.execute(query)
        return result.scalars().first()

    async def get_by_external_id(self, external_id: str) -> Optional[User]:
        """Get user by external ID (from OIDC provider)."""
        query = select(User).where(User.external_id == external_id)
        result = await self.session.execute(query)
        return result.scalars().first()

    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email."""
        query = select(User).where(User.email == email)
        result = await self.session.execute(query)
        return result.scalars().first()

    async def list_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        """List users with pagination."""
        query = select(User).offset(skip).limit(limit)
        result = await self.session.execute(query)
        return result.scalars().all()

    async def create(
            self,
            external_id: str,
            email: str,
            display_name: str,
            role: UserRole = UserRole.VIEWER
    ) -> User:
        """Create a new user."""
        user = User(
            external_id=external_id,
            email=email,
            display_name=display_name,
            role=role
        )
        self.session.add(user)
        await self.session.flush()
        return user

    async def update(self, user: User) -> User:
        """Update an existing user."""
        self.session.add(user)
        await self.session.flush()
        return user

    async def delete(self, user_id: UUID) -> bool:
        """Delete a user by ID."""
        user = await self.get_by_id(user_id)
        if not user:
            return False
        await self.session.delete(user)
        await self.session.flush()
        return True