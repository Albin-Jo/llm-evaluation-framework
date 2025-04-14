import logging
from typing import Dict, List, Optional, Any
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.expression import or_

from backend.app.db.models.orm import Agent, Evaluation
from backend.app.db.repositories.base import BaseRepository

logger = logging.getLogger(__name__)


class AgentRepository(BaseRepository):
    """Repository for Agent model operations."""

    def __init__(self, db: AsyncSession):
        """Initialize repository with Agent model and database session."""
        super().__init__(Agent, db)

    async def get_by_name(self, name: str) -> Optional[Agent]:
        """
        Get an agent by its name.

        Args:
            name: The name of the agent to retrieve

        Returns:
            The agent if found, None otherwise
        """
        query = select(self.model).where(self.model.name == name)
        result = await self.session.execute(query)
        return result.scalars().first()

    async def search_by_name(
            self,
            name_query: str,
            skip: int = 0,
            limit: int = 100,
            additional_filters: Optional[Dict[str, Any]] = None
    ) -> List[Agent]:
        """
        Search for agents with names containing the search query.

        Args:
            name_query: The partial name to search for
            skip: Number of records to skip
            limit: Maximum number of records to return
            additional_filters: Additional filter criteria

        Returns:
            List of agents matching the search criteria
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

    async def get_agents_by_domain(self, domain: str) -> List[Agent]:
        """
        Get all agents for a specific domain.

        Args:
            domain: The domain to filter by

        Returns:
            List of agents in the specified domain
        """
        query = select(self.model).where(self.model.domain == domain)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_active_agents(self) -> List[Agent]:
        """
        Get all active agents.

        Returns:
            List of active agents
        """
        query = select(self.model).where(self.model.is_active == True)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def has_related_evaluations(self, agent_id: UUID) -> bool:
        """
        Check if an agent has any related evaluations.

        Args:
            agent_id: The ID of the agent to check

        Returns:
            True if the agent has evaluations, False otherwise
        """
        query = select(func.count()).select_from(Evaluation).where(Evaluation.agent_id == agent_id)
        result = await self.session.execute(query)
        count = result.scalar()
        return count > 0

    async def search_agents(
            self,
            name: Optional[str] = None,
            domain: Optional[str] = None,
            is_active: Optional[bool] = None,
            tags: Optional[List[str]] = None,
            model_type: Optional[str] = None,
            skip: int = 0,
            limit: int = 100
    ) -> List[Agent]:
        """
        Advanced search for agents with multiple filters.

        Args:
            name: Optional name filter (partial match)
            domain: Optional domain filter
            is_active: Optional active status filter
            tags: Optional list of tags to filter by
            model_type: Optional model type filter
            skip: Number of records to skip
            limit: Maximum number of records to return

        Returns:
            List of agents matching the search criteria
        """
        query = select(self.model)

        # Apply filters
        filters = []

        if name:
            filters.append(self.model.name.ilike(f"%{name}%"))

        if domain:
            filters.append(self.model.domain == domain)

        if is_active is not None:
            filters.append(self.model.is_active == is_active)

        if model_type:
            filters.append(self.model.model_type == model_type)

        # Tags require special handling since they're in a JSON field
        if tags and len(tags) > 0:
            # This implementation depends on your DB specifics
            # For PostgreSQL, you might use the @> operator for JSON contains
            # This is a simplified version that may need adjustment
            for tag in tags:
                # PostgreSQL JSON contains operator
                filters.append(self.model.tags.contains([tag]))

        # Apply all filters if any
        if filters:
            query = query.where(or_(*filters))

        # Add pagination
        query = query.offset(skip).limit(limit)

        # Execute query
        result = await self.session.execute(query)
        return list(result.scalars().all())
