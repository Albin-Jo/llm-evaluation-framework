# File: app/db/repositories/prompt_repository.py
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.db.repositories.base import BaseRepository
from backend.app.db.models.orm.models import Prompt, PromptTemplate, User


class PromptRepository(BaseRepository):
    """Repository for Prompt operations."""

    def __init__(self, db_session: AsyncSession):
        super().__init__(Prompt, db_session)

    async def get_with_template(self, prompt_id: UUID) -> Optional[Prompt]:
        """
        Get a prompt with its template loaded.

        Args:
            prompt_id: Prompt ID

        Returns:
            Optional[Prompt]: Prompt with template or None
        """
        query = (
            select(Prompt)
            .options(selectinload(Prompt.template))
            .where(Prompt.id == prompt_id)
        )
        result = await self.session.execute(query)
        return result.scalars().first()

    async def search_prompts(
            self,
            query: str,
            user: User,
            template_id: Optional[UUID] = None,
            skip: int = 0,
            limit: int = 20
    ) -> List[Prompt]:
        """
        Search for prompts by name, description.

        Args:
            query: Search term
            user: Current user
            template_id: Optional template ID to filter by
            skip: Number of records to skip
            limit: Number of records to return

        Returns:
            List[Prompt]: Matching prompts
        """
        # Base query - filter by name or description
        base_query = (
            select(Prompt)
            .where(
                or_(
                    Prompt.name.ilike(f"%{query}%"),
                    Prompt.description.ilike(f"%{query}%")
                )
            )
        )

        # Filter by template_id if provided
        if template_id:
            base_query = base_query.where(Prompt.template_id == template_id)

        # Add permission filtering
        if user.role.value != "admin":
            # Regular users can see their own prompts and public prompts
            base_query = base_query.where(
                or_(
                    Prompt.owner_id == user.id,
                    Prompt.is_public == True
                )
            )

        # Execute query with pagination
        base_query = base_query.offset(skip).limit(limit)
        result = await self.session.execute(base_query)
        return result.scalars().all()

    async def get_accessible_prompts(
            self,
            user: User,
            template_id: Optional[UUID] = None,
            skip: int = 0,
            limit: int = 100,
            filters: Dict[str, Any] = None
    ) -> List[Prompt]:
        """
        Get prompts accessible to the user (owned or public).

        Args:
            user: Current user
            template_id: Optional template ID to filter by
            skip: Number of records to skip
            limit: Number of records to return
            filters: Additional filters

        Returns:
            List[Prompt]: Accessible prompts
        """
        if filters is None:
            filters = {}

        if template_id:
            filters["template_id"] = template_id

        if user.role.value == "admin":
            # Admins can see all prompts
            return await self.get_multi(skip=skip, limit=limit, filters=filters)

        # For regular users, combine their prompts with public ones
        # Get user's prompts
        user_filters = {**filters, "owner_id": user.id}
        user_prompts = await self.get_multi(skip=0, limit=None, filters=user_filters)

        # If is_public filter is explicitly set to False, don't fetch public prompts
        if filters.get("is_public") is False:
            return user_prompts[skip:skip + limit]

        # Get public prompts
        public_filters = {**filters, "is_public": True}
        public_prompts = await self.get_multi(skip=0, limit=None, filters=public_filters)

        # Filter out user's prompts from public ones to avoid duplicates
        user_prompt_ids = {p.id for p in user_prompts}
        filtered_public_prompts = [p for p in public_prompts if p.id not in user_prompt_ids]

        # Combine and paginate
        all_prompts = user_prompts + filtered_public_prompts
        return all_prompts[skip:skip + limit]


class PromptTemplateRepository(BaseRepository):
    """Repository for PromptTemplate operations."""

    def __init__(self, db_session: AsyncSession):
        super().__init__(PromptTemplate, db_session)

    async def search_templates(
            self,
            query: str,
            is_public: Optional[bool] = None,
            skip: int = 0,
            limit: int = 20
    ) -> List[PromptTemplate]:
        """
        Search for templates by name, description.

        Args:
            query: Search term
            is_public: Optional filter for public status
            skip: Number of records to skip
            limit: Number of records to return

        Returns:
            List[PromptTemplate]: Matching templates
        """
        # Base query - filter by name or description
        base_query = (
            select(PromptTemplate)
            .where(
                or_(
                    PromptTemplate.name.ilike(f"%{query}%"),
                    PromptTemplate.description.ilike(f"%{query}%")
                )
            )
        )

        # Filter by is_public if provided
        if is_public is not None:
            base_query = base_query.where(PromptTemplate.is_public == is_public)

        # Execute query with pagination
        base_query = base_query.offset(skip).limit(limit)
        result = await self.session.execute(base_query)
        return result.scalars().all()