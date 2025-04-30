from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Union
from uuid import UUID

from sqlalchemy import delete, select, update, func, BinaryExpression, or_
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

    async def get_multi(
            self, *, skip: int = 0, limit: int = 100,
            filters: Dict[str, Any] = None,
            load_relationships: List[str] = None,
            sort: Optional[BinaryExpression] = None
    ) -> List[ModelType]:
        """Get multiple records with optional filtering and relationship loading."""
        from sqlalchemy.orm import selectinload

        query = select(self.model)

        # Add filters
        if filters:
            for field, value in filters.items():
                if hasattr(self.model, field):
                    # Handle special case for string fields with LIKE operation
                    if isinstance(value, str) and field not in ["status", "method"]:
                        query = query.where(getattr(self.model, field).ilike(f"%{value}%"))
                    else:
                        query = query.where(getattr(self.model, field) == value)

        # Add relationship loading
        if load_relationships:
            for relationship in load_relationships:
                if hasattr(self.model, relationship):
                    query = query.options(selectinload(getattr(self.model, relationship)))

        # Apply sorting if provided
        if sort:
            query = query.order_by(sort)

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

    async def delete_multi(self, filters: Dict[str, Any] = None) -> int:
        """
        Delete multiple records with filtering.

        Args:
            filters: Filters to apply

        Returns:
            Number of records deleted
        """
        stmt = delete(self.model)

        # Apply filters
        if filters:
            filter_conditions = []
            for field, value in filters.items():
                if hasattr(self.model, field):
                    filter_conditions.append(getattr(self.model, field) == value)

            if filter_conditions:
                stmt = stmt.where(*filter_conditions)

        result = await self.session.execute(stmt)
        return result.rowcount

    async def count(self, filters: Dict[str, Any] = None) -> int:
        """
        Count records with optional filtering.

        Args:
            filters: Optional filters

        Returns:
            int: Count of matching records
        """
        query = select(func.count()).select_from(self.model)

        # Apply filters if provided
        if filters:
            for key, value in filters.items():
                if hasattr(self.model, key):
                    # Handle special case for string fields with LIKE operation
                    if isinstance(value, str) and key not in ["status", "method"]:
                        query = query.where(getattr(self.model, key).ilike(f"%{value}%"))
                    else:
                        query = query.where(getattr(self.model, key) == value)

        result = await self.session.execute(query)
        return result.scalar_one_or_none() or 0