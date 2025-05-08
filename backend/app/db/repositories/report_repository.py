import logging
from typing import Dict, List, Optional, Any
from uuid import UUID

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.expression import or_

from backend.app.db.models.orm import Report, ReportStatus
from backend.app.db.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class ReportRepository(BaseRepository):
    """Repository for Report model operations with user-based access control."""

    def __init__(self, db: AsyncSession):
        """Initialize repository with Report model and database session."""
        super().__init__(Report, db)

    async def get_by_name(self, name: str, user_id: Optional[UUID] = None) -> Optional[Report]:
        """
        Get a report by its name, optionally filtering by user ownership.

        Args:
            name: The name of the report to retrieve
            user_id: Optional user ID to filter by ownership

        Returns:
            The report if found, None otherwise
        """
        query = select(self.model).where(self.model.name == name)

        # Apply user filter if provided
        if user_id:
            query = query.where(self.model.created_by_id == user_id)

        result = await self.session.execute(query)
        return result.scalars().first()

    async def get_by_evaluation(
            self,
            evaluation_id: UUID,
            user_id: Optional[UUID] = None
    ) -> List[Report]:
        """
        Get all reports for a specific evaluation, optionally filtering by user ownership.

        Args:
            evaluation_id: The ID of the evaluation
            user_id: Optional user ID to filter by ownership

        Returns:
            List of reports for the evaluation
        """
        query = select(self.model).where(self.model.evaluation_id == evaluation_id)

        # Apply user filter if provided
        if user_id:
            query = query.where(self.model.created_by_id == user_id)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def search_by_name(
            self,
            name_query: str,
            skip: int = 0,
            limit: int = 100,
            additional_filters: Optional[Dict[str, Any]] = None,
            user_id: Optional[UUID] = None
    ) -> List[Report]:
        """
        Search for reports with names containing the search query,
        optionally filtering by user ownership.

        Args:
            name_query: The partial name to search for
            skip: Number of records to skip
            limit: Maximum number of records to return
            additional_filters: Additional filter criteria
            user_id: Optional user ID to filter by ownership

        Returns:
            List of reports matching the search criteria
        """
        # Start with the LIKE query for the name
        query = select(self.model).where(
            self.model.name.ilike(f"%{name_query}%")
        )

        # Add user filter if provided
        if user_id:
            query = query.where(self.model.created_by_id == user_id)

        # Add any additional filters
        if additional_filters:
            for key, value in additional_filters.items():
                query = query.where(getattr(self.model, key) == value)

        # Add pagination
        query = query.offset(skip).limit(limit)

        # Execute query
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def count_reports_by_status(self, user_id: Optional[UUID] = None) -> Dict[str, int]:
        """
        Count reports grouped by status, optionally filtering by user ownership.

        Args:
            user_id: Optional user ID to filter by ownership

        Returns:
            Dictionary mapping status to count
        """
        query = (
            select(self.model.status, func.count())
            .group_by(self.model.status)
        )

        # Apply user filter if provided
        if user_id:
            query = query.where(self.model.created_by_id == user_id)

        result = await self.session.execute(query)

        # Convert to dictionary
        counts = {status.value: count for status, count in result.all()}

        # Ensure all statuses have a count
        for status in ReportStatus:
            if status.value not in counts:
                counts[status.value] = 0

        return counts

    async def get_public_reports(
            self,
            skip: int = 0,
            limit: int = 100,
            user_id: Optional[UUID] = None
    ) -> List[Report]:
        """
        Get public reports or reports owned by the specified user.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            user_id: Optional user ID to include their private reports

        Returns:
            List of public reports and/or user's reports
        """
        if user_id:
            # Return public reports OR reports owned by this user
            query = (
                select(self.model)
                .where(
                    or_(
                        self.model.is_public == True,
                        self.model.created_by_id == user_id
                    )
                )
                .offset(skip)
                .limit(limit)
            )
        else:
            # Return only public reports
            query = (
                select(self.model)
                .where(self.model.is_public == True)
                .offset(skip)
                .limit(limit)
            )

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_user_report(
            self,
            report_id: UUID,
            user_id: UUID
    ) -> Optional[Report]:
        """
        Get a report by ID that is owned by the specified user.

        Args:
            report_id: The ID of the report to retrieve
            user_id: User ID for ownership check

        Returns:
            The report if found and owned by the user, None otherwise
        """
        query = select(self.model).where(
            and_(
                self.model.id == report_id,
                self.model.created_by_id == user_id
            )
        )

        result = await self.session.execute(query)
        return result.scalars().first()

    async def get_accessible_report(
            self,
            report_id: UUID,
            user_id: Optional[UUID] = None
    ) -> Optional[Report]:
        """
        Get a report by ID that is either public or owned by the specified user.

        Args:
            report_id: The ID of the report to retrieve
            user_id: Optional user ID for ownership check

        Returns:
            The report if found and accessible, None otherwise
        """
        query = select(self.model).where(self.model.id == report_id)

        if user_id:
            # User can access their own reports OR public reports
            query = query.where(
                or_(
                    self.model.is_public == True,
                    self.model.created_by_id == user_id
                )
            )
        else:
            # Anonymous users can only access public reports
            query = query.where(self.model.is_public == True)

        result = await self.session.execute(query)
        return result.scalars().first()
