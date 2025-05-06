from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import select, or_, and_
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.orm import Dataset, DatasetType
from backend.app.db.repositories.base import BaseRepository


class DatasetRepository(BaseRepository):
    """Repository for Dataset operations."""

    def __init__(self, db_session: AsyncSession):
        super().__init__(Dataset, db_session)

    async def search_datasets(
            self,
            query: str,
            types: Optional[List[DatasetType]] = None,
            skip: int = 0,
            limit: int = 20,
            user_id: Optional[UUID] = None
    ) -> List[Dataset]:
        """
        Search for datasets by name, description.
        Returns user's own datasets and public datasets.

        Args:
            query: Search term
            types: Optional list of dataset types to filter by
            skip: Number of records to skip
            limit: Number of records to return
            user_id: ID of the user to filter by ownership

        Returns:
            List[Dataset]: Matching datasets
        """
        # Base query - filter by name or description
        search_condition = or_(
            Dataset.name.ilike(f"%{query}%"),
            Dataset.description.ilike(f"%{query}%")
        )

        # Build access condition: user's own datasets OR public datasets
        if user_id:
            access_condition = or_(
                Dataset.created_by_id == user_id,
                Dataset.is_public == True
            )
            base_condition = and_(search_condition, access_condition)
        else:
            # If no user_id provided, only return public datasets
            base_condition = and_(search_condition, Dataset.is_public == True)

        base_query = select(Dataset).where(base_condition)

        # Filter by types if provided
        if types:
            base_query = base_query.where(Dataset.type.in_([t.value for t in types]))

        # Execute query with pagination
        base_query = base_query.offset(skip).limit(limit)
        result = await self.session.execute(base_query)
        return result.scalars().all()

    async def get_accessible_datasets(
            self,
            user_id: UUID,
            types: Optional[List[DatasetType]] = None,
            skip: int = 0,
            limit: int = 100,
            filters: Dict[str, Any] = None
    ) -> List[Dataset]:
        """
        Get datasets accessible to the user (owned or public).

        Args:
            user_id: ID of the user to filter by ownership
            types: Optional list of dataset types to filter by
            skip: Number of records to skip
            limit: Number of records to return
            filters: Additional filters

        Returns:
            List[Dataset]: Accessible datasets
        """
        if filters is None:
            filters = {}

        # Filter by type if specified
        if types:
            filters["type"] = {"$in": [t.value for t in types]}

        # Build modified filters to include access control
        # User can access their own datasets OR public datasets
        access_condition = or_(
            Dataset.created_by_id == user_id,
            Dataset.is_public == True
        )

        # Start building the query with access control
        query = select(Dataset).where(access_condition)

        # Add other filters
        for field, value in filters.items():
            if hasattr(Dataset, field):
                # Handle special case for string fields with LIKE operation
                if isinstance(value, str) and field not in ["status", "method"]:
                    query = query.where(getattr(Dataset, field).ilike(f"%{value}%"))
                else:
                    query = query.where(getattr(Dataset, field) == value)

        # Add pagination
        query = query.offset(skip).limit(limit)

        # Execute query
        result = await self.session.execute(query)
        return result.scalars().all()

    async def count_accessible_datasets(
            self,
            user_id: UUID,
            filters: Dict[str, Any] = None
    ) -> int:
        """
        Count datasets accessible to the user (owned or public).

        Args:
            user_id: ID of the user to filter by ownership
            filters: Additional filters

        Returns:
            int: Count of accessible datasets
        """
        from sqlalchemy import func

        # Build access condition: user's own datasets OR public datasets
        access_condition = or_(
            Dataset.created_by_id == user_id,
            Dataset.is_public == True
        )

        # Start building query
        query = select(func.count()).select_from(Dataset).where(access_condition)

        # Add other filters
        if filters:
            for field, value in filters.items():
                if hasattr(Dataset, field):
                    # Handle special case for string fields
                    if isinstance(value, str) and field not in ["status", "method"] and field == "name":
                        query = query.where(getattr(Dataset, field).ilike(f"%{value}%"))
                    else:
                        query = query.where(getattr(Dataset, field) == value)

        # Execute query
        result = await self.session.execute(query)
        return result.scalar_one_or_none() or 0

    async def get_user_dataset(
            self,
            dataset_id: UUID,
            user_id: UUID
    ) -> Optional[Dataset]:
        """
        Get a dataset by ID if the user has access (own or public).

        Args:
            dataset_id: Dataset ID
            user_id: User ID for access control

        Returns:
            Optional[Dataset]: Dataset if accessible, None otherwise
        """
        # Build access condition: user's own datasets OR public datasets
        access_condition = or_(
            Dataset.created_by_id == user_id,
            Dataset.is_public == True
        )

        query = select(Dataset).where(
            and_(
                Dataset.id == dataset_id,
                access_condition
            )
        )

        result = await self.session.execute(query)
        return result.scalars().first()

    async def get_owned_dataset(
            self,
            dataset_id: UUID,
            user_id: UUID
    ) -> Optional[Dataset]:
        """
        Get a dataset by ID if the user owns it.

        Args:
            dataset_id: Dataset ID
            user_id: User ID for ownership check

        Returns:
            Optional[Dataset]: Dataset if owned, None otherwise
        """
        query = select(Dataset).where(
            and_(
                Dataset.id == dataset_id,
                Dataset.created_by_id == user_id
            )
        )

        result = await self.session.execute(query)
        return result.scalars().first()
