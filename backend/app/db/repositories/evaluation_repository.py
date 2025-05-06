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
    """Repository for Evaluation model operations."""

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
            sort_dir: str = "desc"
    ) -> List[Evaluation]:
        """
        Search evaluations with text search across name and description.

        Args:
            query_text: Text to search across name and description
            filters: Additional filters to apply
            skip: Number of records to skip
            limit: Maximum number of records to return
            sort_by: Field to sort by
            sort_dir: Sort direction ('asc' or 'desc')

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
            filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """Count evaluations matching search criteria."""
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

        result = await self.session.execute(query)
        return result.scalar() or 0

    async def get_evaluation_with_details(self, evaluation_id: UUID) -> Optional[Evaluation]:
        """
        Get evaluation with all related data in a single query.

        Args:
            evaluation_id: ID of the evaluation to retrieve

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

        result = await self.session.execute(query)
        return result.scalars().first()

    async def get_evaluations_by_status(
            self,
            status: EvaluationStatus,
            skip: int = 0,
            limit: int = 100
    ) -> List[Evaluation]:
        """Get evaluations filtered by status."""
        return await self.get_multi(
            skip=skip,
            limit=limit,
            filters={"status": status},
            load_relationships=["agent", "dataset", "prompt"]
        )

    async def get_evaluations_by_agent(
            self,
            agent_id: UUID,
            skip: int = 0,
            limit: int = 100
    ) -> List[Evaluation]:
        """Get evaluations for a specific agent."""
        return await self.get_multi(
            skip=skip,
            limit=limit,
            filters={"agent_id": agent_id},
            load_relationships=["dataset", "prompt"]
        )

    async def get_evaluations_by_dataset(
            self,
            dataset_id: UUID,
            skip: int = 0,
            limit: int = 100
    ) -> List[Evaluation]:
        """Get evaluations for a specific dataset."""
        return await self.get_multi(
            skip=skip,
            limit=limit,
            filters={"dataset_id": dataset_id},
            load_relationships=["agent", "prompt"]
        )