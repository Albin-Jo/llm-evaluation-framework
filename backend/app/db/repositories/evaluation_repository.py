import logging
from typing import Dict, List, Optional, Any
from uuid import UUID

from sqlalchemy import select, func, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.db.models.orm import Evaluation, EvaluationStatus, EvaluationMethod
from backend.app.db.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class EvaluationRepository(BaseRepository):
    """Repository for Evaluation model operations with user-based access control."""

    def __init__(self, db: AsyncSession):
        """Initialize repository with Evaluation model and database session."""
        super().__init__(Evaluation, db)

    async def search_evaluations(
            self,
            query_text: Optional[str] = None,
            filters: Optional[Dict[str, Any]] = None,
            skip: int = 0,
            limit: int = 100,
            sort_by: str = "created_at",
            sort_dir: str = "desc",
            user_id: Optional[UUID] = None
    ) -> List[Evaluation]:
        """
        Search evaluations with text search across name and description.
        Filter by user ID if provided to limit to user's own evaluations.

        Args:
            query_text: Text to search across name and description
            filters: Additional filters to apply
            skip: Number of records to skip
            limit: Maximum number of records to return
            sort_by: Field to sort by
            sort_dir: Sort direction ('asc' or 'desc')
            user_id: User ID to filter by (if provided)

        Returns:
            List of evaluations matching the search criteria
        """
        query = select(self.model)

        # Text search across multiple fields
        if query_text:
            search_term = f"%{query_text}%"
            text_conditions = or_(
                self.model.name.ilike(search_term),
                self.model.description.ilike(search_term)
            )
            query = query.where(text_conditions)

        # Apply additional filters
        if filters:
            filter_conditions = []
            for key, value in filters.items():
                if hasattr(self.model, key):
                    filter_conditions.append(getattr(self.model, key) == value)

            if filter_conditions:
                query = query.where(and_(*filter_conditions))

        # Filter by user ID if provided
        if user_id:
            query = query.where(self.model.created_by_id == user_id)

        # Add relationships to load eagerly
        query = query.options(
            selectinload(self.model.agent),
            selectinload(self.model.dataset),
            selectinload(self.model.prompt)
        )

        # Apply sorting
        sort_column = getattr(self.model, sort_by, self.model.created_at)
        if sort_dir.lower() == "asc":
            query = query.order_by(sort_column.asc())
        else:
            query = query.order_by(sort_column.desc())

        # Add pagination
        query = query.offset(skip).limit(limit)

        # Execute query
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def count_search_evaluations(
            self,
            query_text: Optional[str] = None,
            filters: Optional[Dict[str, Any]] = None,
            user_id: Optional[UUID] = None
    ) -> int:
        """
        Count evaluations matching search criteria.
        Filter by user ID if provided to limit to user's own evaluations.

        Args:
            query_text: Text to search across name and description
            filters: Additional filters to apply
            user_id: User ID to filter by (if provided)

        Returns:
            Count of matching evaluations
        """
        query = select(func.count()).select_from(self.model)

        # Text search across multiple fields
        if query_text:
            search_term = f"%{query_text}%"
            text_conditions = or_(
                self.model.name.ilike(search_term),
                self.model.description.ilike(search_term)
            )
            query = query.where(text_conditions)

        # Apply additional filters
        if filters:
            filter_conditions = []
            for key, value in filters.items():
                if hasattr(self.model, key):
                    filter_conditions.append(getattr(self.model, key) == value)

            if filter_conditions:
                query = query.where(and_(*filter_conditions))

        # Filter by user ID if provided
        if user_id:
            query = query.where(self.model.created_by_id == user_id)

        result = await self.session.execute(query)
        return result.scalar() or 0

    async def get_evaluation_with_details(
            self,
            evaluation_id: UUID,
            user_id: Optional[UUID] = None
    ) -> Optional[Evaluation]:
        """
        Get evaluation with all related data in a single query.
        Optionally filter by user ID to ensure ownership.

        Args:
            evaluation_id: ID of the evaluation to retrieve
            user_id: Optional user ID to filter by ownership

        Returns:
            Evaluation with all relationships loaded
        """
        query = (
            select(self.model)
            .options(
                selectinload(self.model.agent),
                selectinload(self.model.dataset),
                selectinload(self.model.prompt),
                selectinload(self.model.results)
                .selectinload(self.model.results.property.mapper.class_.metric_scores)
            )
            .where(self.model.id == evaluation_id)
        )

        # Add user filter if requested
        if user_id:
            query = query.where(self.model.created_by_id == user_id)

        result = await self.session.execute(query)
        return result.scalars().first()

    async def get_user_evaluations(
            self,
            user_id: UUID,
            status: Optional[EvaluationStatus] = None,
            skip: int = 0,
            limit: int = 100
    ) -> List[Evaluation]:
        """
        Get evaluations for a specific user with optional status filter.

        Args:
            user_id: User ID to filter by
            status: Optional status filter
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of evaluations owned by the user
        """
        filters = {"created_by_id": user_id}
        if status:
            filters["status"] = status

        return await self.get_multi(
            skip=skip,
            limit=limit,
            filters=filters,
            load_relationships=["agent", "dataset", "prompt"]
        )

    async def get_user_evaluation(
            self,
            evaluation_id: UUID,
            user_id: UUID
    ) -> Optional[Evaluation]:
        """
        Get a specific evaluation owned by a user.

        Args:
            evaluation_id: Evaluation ID to retrieve
            user_id: User ID for ownership check

        Returns:
            Evaluation if owned by the user, None otherwise
        """
        query = select(self.model).where(
            and_(
                self.model.id == evaluation_id,
                self.model.created_by_id == user_id
            )
        )

        result = await self.session.execute(query)
        return result.scalars().first()