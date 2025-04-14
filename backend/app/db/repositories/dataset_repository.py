from typing import List, Optional, Dict, Any

from sqlalchemy import select, or_
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
            limit: int = 20
    ) -> List[Dataset]:
        """
        Search for datasets by name, description.

        Args:
            query: Search term
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

        # Execute query with pagination
        base_query = base_query.offset(skip).limit(limit)
        result = await self.db_session.execute(base_query)
        return result.scalars().all()

    async def get_accessible_datasets(
            self,
            types: Optional[List[DatasetType]] = None,
            skip: int = 0,
            limit: int = 100,
            filters: Dict[str, Any] = None
    ) -> List[Dataset]:
        """
        Get datasets accessible to the user (owned or public).
        Args:
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
        type_filter = {}
        if types:
            type_filter = {"type": {"$in": [t.value for t in types]}}

        # Combine all filters
        combined_filters = {**filters, **type_filter}

        # Get all datasets based on combined filters without applying is_public filter
        # This will get both public and private datasets matching the filters
        all_datasets = await self.get_multi(skip=skip, limit=limit, filters=combined_filters)

        print(f"Retrieved {len(all_datasets)} datasets matching filters: {combined_filters}")

        return all_datasets
