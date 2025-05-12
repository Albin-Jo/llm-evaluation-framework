# File: backend/app/db/repositories/agent_repository.py
import logging
from typing import Dict, List, Optional, Any
from uuid import UUID

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.expression import or_

from backend.app.db.models.orm import Agent, Evaluation
from backend.app.db.repositories.base import BaseRepository
from backend.app.utils.credential_utils import encrypt_credentials, decrypt_credentials, mask_credentials

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

    async def count_with_search(self, name_query: str, additional_filters: Optional[Dict[str, Any]] = None) -> int:
        """Count agents matching search criteria."""
        query = select(func.count()).select_from(self.model).where(
            self.model.name.ilike(f"%{name_query}%")
        )

        if additional_filters:
            for key, value in additional_filters.items():
                query = query.where(getattr(self.model, key) == value)

        result = await self.session.execute(query)
        return result.scalar() or 0

    async def get_agents_by_domain(self, domain: str, user_id: Optional[UUID] = None) -> List[Agent]:
        """
        Get all agents for a specific domain, optionally filtered by user.

        Args:
            domain: The domain to filter by
            user_id: Optional user ID to filter by ownership

        Returns:
            List of agents in the specified domain
        """
        query = select(self.model).where(self.model.domain == domain)

        # Add user filter if provided
        if user_id:
            query = query.where(self.model.created_by_id == user_id)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_active_agents(self, user_id: Optional[UUID] = None) -> List[Agent]:
        """
        Get all active agents, optionally filtered by user.

        Args:
            user_id: Optional user ID to filter by ownership

        Returns:
            List of active agents
        """
        query = select(self.model).where(self.model.is_active == True)

        # Add user filter if provided
        if user_id:
            query = query.where(self.model.created_by_id == user_id)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_user_agents(self, user_id: UUID) -> List[Agent]:
        """
        Get all agents owned by a specific user.

        Args:
            user_id: User ID to filter by

        Returns:
            List of agents owned by the user
        """
        query = select(self.model).where(self.model.created_by_id == user_id)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_user_agent(self, agent_id: UUID, user_id: UUID) -> Optional[Agent]:
        """
        Get an agent by ID if owned by a specific user.

        Args:
            agent_id: Agent ID
            user_id: User ID to check ownership

        Returns:
            Agent if found and owned by user, None otherwise
        """
        query = select(self.model).where(
            and_(
                self.model.id == agent_id,
                self.model.created_by_id == user_id
            )
        )
        result = await self.session.execute(query)
        return result.scalars().first()

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

    # Methods for handling encrypted credentials

    async def create_with_encrypted_credentials(self, data: Dict[str, Any]) -> Agent:
        """
        Create an agent with encrypted credentials.

        Args:
            data: Agent data

        Returns:
            Created agent
        """
        # Encrypt credentials if present
        if data.get('auth_credentials'):
            # Log masked credentials for debugging
            masked_creds = mask_credentials(data['auth_credentials'])
            logger.debug(f"Encrypting credentials: {masked_creds}")

            # Store encrypted version
            data['auth_credentials'] = encrypt_credentials(data['auth_credentials'])

        # Create agent
        return await self.create(data)

    async def update_with_encrypted_credentials(self, id: UUID, data: Dict[str, Any]) -> Optional[Agent]:
        """
        Update an agent with encrypted credentials.

        Args:
            id: Agent ID
            data: Update data

        Returns:
            Updated agent
        """
        # Encrypt credentials if present
        if 'auth_credentials' in data and data['auth_credentials'] is not None:
            # Log masked credentials for debugging
            masked_creds = mask_credentials(data['auth_credentials'])
            logger.debug(f"Encrypting credentials for update: {masked_creds}")

            data['auth_credentials'] = encrypt_credentials(data['auth_credentials'])

        # Update agent
        return await self.update(id, data)

    async def get_with_decrypted_credentials(self, id: UUID) -> Optional[Agent]:
        """
        Get agent with decrypted credentials.

        Args:
            id: Agent ID

        Returns:
            Agent with decrypted credentials
        """
        agent = await self.get(id)
        if not agent:
            return None

        # Decrypt credentials if present
        if agent.auth_credentials:
            try:
                # Check if credentials are already a dictionary (not encrypted)
                if isinstance(agent.auth_credentials, dict):
                    logger.debug(f"Credentials for agent {id} are already decrypted")
                    return agent

                # Decrypt the credentials
                agent.auth_credentials = decrypt_credentials(agent.auth_credentials)
                logger.debug(f"Decrypted credentials for agent {id}")
            except Exception as e:
                logger.error(f"Error decrypting credentials for agent {id}: {e}")
                # Set to None if decryption fails
                agent.auth_credentials = None

        return agent

    async def get_user_owned_with_decrypted_credentials(self, id: UUID, user_id: UUID) -> Optional[Agent]:
        """
        Get agent with decrypted credentials if owned by user.

        Args:
            id: Agent ID
            user_id: User ID to check ownership

        Returns:
            Agent with decrypted credentials if owned by user, None otherwise
        """
        # First check if the user owns this agent
        query = select(self.model).where(
            and_(
                self.model.id == id,
                self.model.created_by_id == user_id
            )
        )
        result = await self.session.execute(query)
        agent = result.scalars().first()

        if not agent:
            return None

        # Decrypt credentials if present
        if agent.auth_credentials:
            try:
                # Check if credentials are already a dictionary (not encrypted)
                if isinstance(agent.auth_credentials, dict):
                    logger.debug(f"Credentials for agent {id} are already decrypted")
                    return agent

                # Decrypt the credentials
                agent.auth_credentials = decrypt_credentials(agent.auth_credentials)
                logger.debug(f"Decrypted credentials for agent {id}")
            except Exception as e:
                logger.error(f"Error decrypting credentials for agent {id}: {e}")
                # Set to None if decryption fails
                agent.auth_credentials = None

        return agent

    async def search_agents(
            self,
            name: Optional[str] = None,
            domain: Optional[str] = None,
            is_active: Optional[bool] = None,
            tags: Optional[List[str]] = None,
            model_type: Optional[str] = None,
            user_id: Optional[UUID] = None,
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
            user_id: Optional user ID to filter by ownership
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

        if user_id:
            filters.append(self.model.created_by_id == user_id)

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
            query = query.where(and_(*filters))

        # Add pagination
        query = query.offset(skip).limit(limit)

        # Execute query
        result = await self.session.execute(query)
        return list(result.scalars().all())