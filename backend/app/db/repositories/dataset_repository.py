# File: app/db/repositories/dataset_repository.py
from typing import List, Optional, Dict, Any
from uuid import UUID

from sqlalchemy import select, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.base import BaseRepository
from app.models.orm.models import Dataset, DatasetType, User


class DatasetRepository(BaseRepository):
    """Repository for Dataset operations."""

    def __init__(self, db_session: AsyncSession):
        super().__init__(Dataset, db_session)

    async def search_datasets(
            self,
            query: str,
            user: User,
            types: Optional[List[DatasetType]] = None,
            skip: int = 0,
            limit: int = 20
    ) -> List[Dataset]:
        """
        Search for datasets by name, description.

        Args:
            query: Search term
            user: Current user
            types: Optional list of dataset types to filter by
            skip: Number of records to skip
            limit: Number of records to return

        Returns:
            List[Dataset]: Matching datasets
        """
        # Base query - filter by name or description
        base_query = (
            select(Dataset)
            .where(
                or_(
                    Dataset.name.ilike(f"%{query}%"),
                    Dataset.description.ilike(f"%{query}%")
                )
            )
        )

        # Filter by types if provided
        if types:
            base_query = base_query.where(Dataset.type.in_([t.value for t in types]))

        # Add permission filtering
        if user.role.value != "admin":
            # Regular users can see their own datasets and public datasets
            base_query = base_query.where(
                or_(
                    Dataset.owner_id == user.id,
                    Dataset.is_public == True
                )
            )

        # Execute query with pagination
        base_query = base_query.offset(skip).limit(limit)
        result = await self.db_session.execute(base_query)
        return result.scalars().all()

    async def get_accessible_datasets(
            self,
            user: User,
            types: Optional[List[DatasetType]] = None,
            skip: int = 0,
            limit: int = 100,
            filters: Dict[str, Any] = None
    ) -> List[Dataset]:
        """
        Get datasets accessible to the user (owned or public).

        Args:
            user: Current user
            types: Optional list of dataset types to filter by
            skip: Number of records to skip
            limit: Number of records to return
            filters: Additional filters

        Returns:
            List[Dataset]: Accessible datasets
        """
        if filters is None:
            filters = {}

        if user.role.value == "admin":
            # Admins can see all datasets
            return await self.get_multi(skip=skip, limit=limit, filters=filters)

        # For regular users, combine their datasets with public ones
        # Filter by type if specified
        type_filter = {}
        if types:
            type_filter = {"type": {"$in": [t.value for t in types]}}

        # Get user's datasets
        user_filters = {**filters, "owner_id": user.id, **type_filter}
        user_datasets = await self.get_multi(skip=0, limit=None, filters=user_filters)

        # If is_public filter is explicitly set to False, don't fetch public datasets
        if filters.get("is_public") is False:
            return user_datasets[skip:skip + limit]

        # Get public datasets
        public_filters = {**filters, "is_public": True, **type_filter}
        public_datasets = await self.get_multi(skip=0, limit=None, filters=public_filters)

        # Filter out user's datasets from public ones to avoid duplicates
        user_dataset_ids = {ds.id for ds in user_datasets}
        filtered_public_datasets = [ds for ds in public_datasets if ds.id not in user_dataset_ids]

        # Combine and paginate
        all_datasets = user_datasets + filtered_public_datasets
        return all_datasets[skip:skip + limit]