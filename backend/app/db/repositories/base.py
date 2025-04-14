from typing import Any, Dict, Generic, List, Optional, Type, TypeVar
from uuid import UUID

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.orm import Base

# Define a TypeVar for models
ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Base repository for common database operations."""

    def __init__(self, model: Type[ModelType], session: AsyncSession):
        self.model = model
        self.session = session

    async def get(self, id: UUID) -> Optional[ModelType]:
        """Get a record by ID."""
        query = select(self.model).where(self.model.id == id)
        result = await self.session.execute(query)
        return result.scalars().first()

    # Optimize get_multi in base.py to use selectinload for relationships
    async def get_multi(
            self, *, skip: int = 0, limit: int = 100,
            filters: Dict[str, Any] = None,
            load_relationships: List[str] = None
    ) -> List[ModelType]:
        """Get multiple records with optional filtering and relationship loading."""
        from sqlalchemy.orm import selectinload

        query = select(self.model)

        # Add filters
        if filters:
            for field, value in filters.items():
                if hasattr(self.model, field):
                    query = query.where(getattr(self.model, field) == value)

        # Add relationship loading
        if load_relationships:
            for relationship in load_relationships:
                if hasattr(self.model, relationship):
                    query = query.options(selectinload(getattr(self.model, relationship)))

        # Add pagination
        query = query.offset(skip).limit(limit)

        # Execute query
        result = await self.session.execute(query)
        return result.scalars().all()

    async def create(self, obj_in: Dict[str, Any]) -> ModelType:
        """
        Create a new record.

        Args:
            obj_in: Dictionary of field-value pairs

        Returns:
            Created model instance
        """
        db_obj = self.model(**obj_in)
        self.session.add(db_obj)
        await self.session.flush()
        return db_obj

    async def update(self, id: UUID, obj_in: Dict[str, Any]) -> Optional[ModelType]:
        """
        Update a record by ID.

        Args:
            id: ID of the record to update
            obj_in: Dictionary of field-value pairs to update

        Returns:
            Updated model instance or None if not found
        """
        stmt = (
            update(self.model)
            .where(self.model.id == id)
            .values(**obj_in)
            .returning(self.model)
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def delete(self, id: UUID) -> bool:
        """
        Delete a record by ID.

        Args:
            id: ID of the record to delete

        Returns:
            True if record was deleted, False if not found
        """
        stmt = delete(self.model).where(self.model.id == id)
        result = await self.session.execute(stmt)
        return result.rowcount > 0
