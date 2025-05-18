import logging
from typing import Dict, List, Optional, Any
from uuid import UUID

from sqlalchemy import select, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.db.models.orm import EvaluationComparison
from backend.app.db.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class ComparisonRepository(BaseRepository):
    """Repository for EvaluationComparison model operations with user-based access control."""

    def __init__(self, db: AsyncSession):
        """Initialize repository with EvaluationComparison model and database session."""
        super().__init__(EvaluationComparison, db)

    async def get_with_evaluations(self, comparison_id: UUID) -> Optional[EvaluationComparison]:
        """
        Get comparison with related evaluations loaded.

        Args:
            comparison_id: The ID of the comparison to retrieve

        Returns:
            Optional[EvaluationComparison]: Comparison with loaded relationships
        """
        query = (
            select(self.model)
            .options(
                selectinload(self.model.evaluation_a),
                selectinload(self.model.evaluation_b)
            )
            .where(self.model.id == comparison_id)
        )

        result = await self.session.execute(query)
        return result.scalars().first()

    async def get_by_evaluations(
            self,
            evaluation_a_id: UUID,
            evaluation_b_id: UUID,
            user_id: Optional[UUID] = None
    ) -> Optional[EvaluationComparison]:
        """
        Find a comparison by the two evaluation IDs it references.

        Args:
            evaluation_a_id: First evaluation ID
            evaluation_b_id: Second evaluation ID
            user_id: Optional user ID for ownership check

        Returns:
            Optional[EvaluationComparison]: Comparison if found, None otherwise
        """
        query = (
            select(self.model)
            .where(
                or_(
                    and_(
                        self.model.evaluation_a_id == evaluation_a_id,
                        self.model.evaluation_b_id == evaluation_b_id
                    ),
                    and_(
                        self.model.evaluation_a_id == evaluation_b_id,
                        self.model.evaluation_b_id == evaluation_a_id
                    )
                )
            )
        )

        # Add user filter if provided
        if user_id:
            query = query.where(self.model.created_by_id == user_id)

        result = await self.session.execute(query)
        return result.scalars().first()

    async def search_comparisons(
            self,
            query_text: Optional[str] = None,
            filters: Optional[Dict[str, Any]] = None,
            skip: int = 0,
            limit: int = 100,
            sort_by: str = "created_at",
            sort_dir: str = "desc",
            user_id: Optional[UUID] = None
    ) -> List[EvaluationComparison]:
        """
        Search comparisons with text search across name and description.
        Filter by user ID if provided to limit to user's own comparisons.

        Args:
            query_text: Text to search across name and description
            filters: Additional filters to apply
            skip: Number of records to skip
            limit: Maximum number of records to return
            sort_by: Field to sort by
            sort_dir: Sort direction ('asc' or 'desc')
            user_id: User ID to filter by (if provided)

        Returns:
            List[EvaluationComparison]: List of comparisons matching criteria
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
            selectinload(self.model.evaluation_a),
            selectinload(self.model.evaluation_b),
            selectinload(self.model.created_by)
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

    async def count_search_comparisons(
            self,
            query_text: Optional[str] = None,
            filters: Optional[Dict[str, Any]] = None,
            user_id: Optional[UUID] = None
    ) -> int:
        """
        Count comparisons matching search criteria.

        Args:
            query_text: Text to search across name and description
            filters: Additional filters to apply
            user_id: User ID to filter by (if provided)

        Returns:
            Count of matching comparisons
        """
        from sqlalchemy import func

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

    async def get_user_comparisons(
            self,
            user_id: UUID,
            skip: int = 0,
            limit: int = 100
    ) -> List[EvaluationComparison]:
        """
        Get comparisons for a specific user.

        Args:
            user_id: User ID to filter by
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of comparisons owned by the user
        """
        return await self.get_multi(
            skip=skip,
            limit=limit,
            filters={"created_by_id": user_id},
            load_relationships=["evaluation_a", "evaluation_b", "created_by"]
        )
