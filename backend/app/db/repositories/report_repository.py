# File: backend/app/db/repositories/report_repository.py
import logging
from typing import Dict, List, Optional, Any
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.expression import or_

from backend.app.db.models.orm import Report, ReportStatus
from backend.app.db.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class ReportRepository(BaseRepository):
    """Repository for Report model operations."""

    def __init__(self, db: AsyncSession):
        """Initialize repository with Report model and database session."""
        super().__init__(Report, db)

    async def get_by_name(self, name: str) -> Optional[Report]:
        """
        Get a report by its name.

        Args:
            name: The name of the report to retrieve

        Returns:
            The report if found, None otherwise
        """
        query = select(self.model).where(self.model.name == name)
        result = await self.session.execute(query)
        return result.scalars().first()

    async def get_by_evaluation(self, evaluation_id: UUID) -> List[Report]:
        """
        Get all reports for a specific evaluation.

        Args:
            evaluation_id: The ID of the evaluation

        Returns:
            List of reports for the evaluation
        """
        query = select(self.model).where(self.model.evaluation_id == evaluation_id)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def search_by_name(
            self,
            name_query: str,
            skip: int = 0,
            limit: int = 100,
            additional_filters: Optional[Dict[str, Any]] = None
    ) -> List[Report]:
        """
        Search for reports with names containing the search query.

        Args:
            name_query: The partial name to search for
            skip: Number of records to skip
            limit: Maximum number of records to return
            additional_filters: Additional filter criteria

        Returns:
            List of reports matching the search criteria
        """
        # Start with the LIKE query for the name
        query = select(self.model).where(
            self.model.name.ilike(f"%{name_query}%")
        )

        # Add any additional filters
        if additional_filters:
            for key, value in additional_filters.items():
                query = query.where(getattr(self.model, key) == value)

        # Add pagination
        query = query.offset(skip).limit(limit)

        # Execute query
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def count_reports_by_status(self) -> Dict[str, int]:
        """
        Count reports grouped by status.

        Returns:
            Dictionary mapping status to count
        """
        query = (
            select(self.model.status, func.count())
            .group_by(self.model.status)
        )
        result = await self.session.execute(query)

        # Convert to dictionary
        counts = {status.value: count for status, count in result.all()}

        # Ensure all statuses have a count
        for status in ReportStatus:
            if status.value not in counts:
                counts[status.value] = 0

        return counts

    async def get_public_reports(
            self, skip: int = 0, limit: int = 100
    ) -> List[Report]:
        """
        Get all public reports.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of public reports
        """
        query = (
            select(self.model)
            .where(self.model.is_public == True)
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(query)
        return list(result.scalars().all())