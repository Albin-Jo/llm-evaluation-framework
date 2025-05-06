from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.orm import User, UserRole


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

    async def search_users(self, query: str, skip: int = 0, limit: int = 100) -> List[User]:
        """
        Search for users by email or display name.

        Args:
            query: Search term
            skip: Number of records to skip
            limit: Maximum records to return

        Returns:
            List of matching users
        """
        search_query = select(User).where(
            or_(
                User.email.ilike(f"%{query}%"),
                User.display_name.ilike(f"%{query}%")
            )
        ).offset(skip).limit(limit)

        result = await self.session.execute(search_query)
        return result.scalars().all()

    async def sync_user_from_token(
            self,
            external_id: str,
            email: str,
            display_name: str,
            roles: List[str] = None
    ) -> User:
        """
        Sync user from token data. Creates if not exists, updates if exists.

        Args:
            external_id: External ID from OIDC provider
            email: User email
            display_name: User display name
            roles: List of roles from token

        Returns:
            User: Updated or created user
        """
        # Check if user exists
        user = await self.get_by_external_id(external_id)

        # Determine role from token roles
        role = UserRole.VIEWER
        if roles:
            if "admin" in roles:
                role = UserRole.ADMIN
            elif "evaluator" in roles:
                role = UserRole.EVALUATOR

        if user:
            # Update existing user if needed
            updated = False

            if user.email != email:
                user.email = email
                updated = True

            if user.display_name != display_name:
                user.display_name = display_name
                updated = True

            # Only update role if it's higher privilege than current
            role_priority = {
                UserRole.VIEWER: 1,
                UserRole.EVALUATOR: 2,
                UserRole.ADMIN: 3
            }

            if role_priority.get(role, 0) > role_priority.get(user.role, 0):
                user.role = role
                updated = True

            if updated:
                self.session.add(user)
                await self.session.flush()

            return user
        else:
            # Create new user
            return await self.create(
                external_id=external_id,
                email=email,
                display_name=display_name,
                role=role
            )