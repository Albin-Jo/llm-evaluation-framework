from typing import Any, Dict, Generic, List, Optional, Type, TypeVar, Tuple
from uuid import UUID

from sqlalchemy import delete, select, update, func, BinaryExpression, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.db.models.orm import Base

# Define a TypeVar for models
ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """Base repository for common database operations with user-based access control."""

    def __init__(self, model: Type[ModelType], session: AsyncSession):
        self.model = model
        self.session = session

    async def get(self, id: UUID) -> Optional[ModelType]:
        """Get a record by ID."""
        query = select(self.model).where(self.model.id == id)
        result = await self.session.execute(query)
        return result.scalars().first()

    async def get_user_owned(self, id: UUID, user_id: UUID) -> Optional[ModelType]:
        """
        Get a record by ID that is owned by the specified user.

        Args:
            id: ID of the record to retrieve
            user_id: User ID for ownership check

        Returns:
            The record if found and owned by the user, None otherwise
        """
        # Check if the model has created_by_id attribute
        if not hasattr(self.model, "created_by_id"):
            raise AttributeError(f"Model {self.model.__name__} does not have created_by_id attribute")

        query = select(self.model).where(
            and_(
                self.model.id == id,
                self.model.created_by_id == user_id
            )
        )
        result = await self.session.execute(query)
        return result.scalars().first()

    async def get_accessible(self, id: UUID, user_id: Optional[UUID] = None) -> Optional[ModelType]:
        """
        Get a record by ID that is either public or owned by the specified user.

        Args:
            id: ID of the record to retrieve
            user_id: Optional user ID for ownership check

        Returns:
            The record if found and accessible, None otherwise
        """
        # Check if the model has is_public and created_by_id attributes
        has_public = hasattr(self.model, "is_public")
        has_owner = hasattr(self.model, "created_by_id")

        if not (has_public and has_owner):
            # Fall back to regular get if model doesn't support access control
            return await self.get(id)

        query = select(self.model).where(self.model.id == id)

        if user_id:
            # User can access their own records OR public records
            query = query.where(
                or_(
                    self.model.is_public == True,
                    self.model.created_by_id == user_id
                )
            )
        else:
            # Anonymous users can only access public records
            query = query.where(self.model.is_public == True)

        result = await self.session.execute(query)
        return result.scalars().first()

    async def get_multi(
            self, *,
            skip: int = 0,
            limit: int = 100,
            filters: Dict[str, Any] = None,
            load_relationships: List[str] = None,
            sort: Optional[BinaryExpression] = None,
            user_id: Optional[UUID] = None
    ) -> List[ModelType]:
        """
        Get multiple records with optional filtering, relationship loading, and user access control.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: Optional filters to apply
            load_relationships: Optional list of relationships to load
            sort: Optional sort expression
            user_id: Optional user ID to filter by ownership

        Returns:
            List of records matching the criteria
        """
        query = select(self.model)

        # Add user filter if provided and the model has created_by_id
        if user_id is not None and hasattr(self.model, "created_by_id"):
            # Check if model has is_public field for public/private filtering
            if hasattr(self.model, "is_public"):
                # Show user's records or public records
                query = query.where(
                    or_(
                        self.model.created_by_id == user_id,
                        self.model.is_public == True
                    )
                )
            else:
                # Just filter by ownership
                query = query.where(self.model.created_by_id == user_id)

        # Add filters
        filter_conditions = []
        if filters:
            for field, value in filters.items():
                if hasattr(self.model, field):
                    # Handle special case for string fields with LIKE operation
                    if isinstance(value, str) and field not in ["status", "method"]:
                        filter_conditions.append(getattr(self.model, field).ilike(f"%{value}%"))
                    else:
                        filter_conditions.append(getattr(self.model, field) == value)

        # Apply filter conditions if any
        if filter_conditions:
            query = query.where(and_(*filter_conditions))

        # Add relationship loading
        if load_relationships:
            for relationship in load_relationships:
                if hasattr(self.model, relationship):
                    query = query.options(selectinload(getattr(self.model, relationship)))

        # Apply sorting if provided
        if sort is not None:
            query = query.order_by(sort)

        # Add pagination
        query = query.offset(skip).limit(limit)

        # Execute query
        result = await self.session.execute(query)
        return result.scalars().all()

    async def get_multi_with_count(
            self, *,
            skip: int = 0,
            limit: int = 100,
            filters: Dict[str, Any] = None,
            load_relationships: List[str] = None,
            sort: Optional[BinaryExpression] = None,
            user_id: Optional[UUID] = None
    ) -> Tuple[List[ModelType], int]:
        """
        Get multiple records with total count in a single query,
        with optional user access control.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            filters: Optional filters to apply
            load_relationships: Optional list of relationships to load
            sort: Optional sort expression
            user_id: Optional user ID to filter by ownership

        Returns:
            Tuple containing the list of records and total count
        """
        # Base query
        query = select(self.model)

        # Add user filter if provided and the model has created_by_id
        if user_id is not None and hasattr(self.model, "created_by_id"):
            # Check if model has is_public field for public/private filtering
            if hasattr(self.model, "is_public"):
                # Show user's records or public records
                query = query.where(
                    or_(
                        self.model.created_by_id == user_id,
                        self.model.is_public == True
                    )
                )
            else:
                # Just filter by ownership
                query = query.where(self.model.created_by_id == user_id)

        # Add filters
        filter_conditions = []
        if filters:
            for field, value in filters.items():
                if hasattr(self.model, field):
                    # Handle special case for string fields with LIKE operation
                    if isinstance(value, str) and field not in ["status", "method"]:
                        filter_conditions.append(getattr(self.model, field).ilike(f"%{value}%"))
                    else:
                        filter_conditions.append(getattr(self.model, field) == value)

        # Apply filter conditions if any
        if filter_conditions:
            query = query.where(and_(*filter_conditions))

        # Add count using window function
        count_query = query.add_columns(func.count().over().label('total_count'))

        # Add relationship loading
        if load_relationships:
            for relationship in load_relationships:
                if hasattr(self.model, relationship):
                    count_query = count_query.options(selectinload(getattr(self.model, relationship)))

        # Apply sorting if provided
        if sort is not None:
            count_query = count_query.order_by(sort)

        # Add pagination
        count_query = count_query.offset(skip).limit(limit)

        # Execute query
        result = await self.session.execute(count_query)
        rows = result.all()

        if not rows:
            return [], 0

        # Extract models and count
        models = [row[0] for row in rows]
        total_count = rows[0].total_count if rows else 0

        return models, total_count

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

    async def update(
            self,
            id: UUID,
            obj_in: Dict[str, Any],
            user_id: Optional[UUID] = None
    ) -> Optional[ModelType]:
        """
        Update a record by ID with optional user ownership verification.

        Args:
            id: ID of the record to update
            obj_in: Dictionary of field-value pairs to update
            user_id: Optional user ID for ownership verification

        Returns:
            Updated model instance or None if not found or user doesn't have access
        """
        # Check ownership if user_id provided and model has created_by_id
        if user_id is not None and hasattr(self.model, "created_by_id"):
            record = await self.get_user_owned(id, user_id)
            if not record:
                return None

        stmt = (
            update(self.model)
            .where(self.model.id == id)
            .values(**obj_in)
            .returning(self.model)
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def delete(
            self,
            id: UUID,
            user_id: Optional[UUID] = None
    ) -> bool:
        """
        Delete a record by ID with optional user ownership verification.

        Args:
            id: ID of the record to delete
            user_id: Optional user ID for ownership verification

        Returns:
            True if record was deleted, False if not found or user doesn't have access
        """
        # Check ownership if user_id provided and model has created_by_id
        if user_id is not None and hasattr(self.model, "created_by_id"):
            record = await self.get_user_owned(id, user_id)
            if not record:
                return False

        stmt = delete(self.model).where(self.model.id == id)
        result = await self.session.execute(stmt)
        return result.rowcount > 0

    async def delete_multi(
            self,
            filters: Dict[str, Any] = None,
            user_id: Optional[UUID] = None
    ) -> int:
        """
        Delete multiple records with filtering and optional user ownership verification.

        Args:
            filters: Filters to apply
            user_id: Optional user ID for ownership verification

        Returns:
            Number of records deleted
        """
        stmt = delete(self.model)

        # Apply user filter if provided and model has created_by_id
        if user_id is not None and hasattr(self.model, "created_by_id"):
            stmt = stmt.where(self.model.created_by_id == user_id)

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

    async def count(
            self,
            filters: Dict[str, Any] = None,
            user_id: Optional[UUID] = None
    ) -> int:
        """
        Count records with optional filtering and user access control.

        Args:
            filters: Optional filters
            user_id: Optional user ID for ownership filtering

        Returns:
            int: Count of matching records
        """
        query = select(func.count()).select_from(self.model)

        # Add user filter if provided and model has created_by_id
        if user_id is not None and hasattr(self.model, "created_by_id"):
            # Check if model has is_public field for public/private filtering
            if hasattr(self.model, "is_public"):
                # Count user's records or public records
                query = query.where(
                    or_(
                        self.model.created_by_id == user_id,
                        self.model.is_public == True
                    )
                )
            else:
                # Just filter by ownership
                query = query.where(self.model.created_by_id == user_id)

        # Apply filters if provided
        filter_conditions = []
        if filters:
            for key, value in filters.items():
                if hasattr(self.model, key):
                    # Handle special case for string fields - use exact match for count
                    # to match the main query behavior
                    if isinstance(value, str) and key not in ["status", "method"] and key == "name":
                        filter_conditions.append(getattr(self.model, key).ilike(f"%{value}%"))
                    else:
                        filter_conditions.append(getattr(self.model, key) == value)

        # Apply filter conditions if any
        if filter_conditions:
            query = query.where(and_(*filter_conditions))

        result = await self.session.execute(query)
        return result.scalar_one_or_none() or 0

    async def exists(
            self,
            filters: Dict[str, Any],
            user_id: Optional[UUID] = None
    ) -> bool:
        """
        Check if a record exists matching the filters with optional user access control.

        Args:
            filters: Filter criteria
            user_id: Optional user ID for ownership filtering

        Returns:
            bool: True if exists, False otherwise
        """
        query = select(func.count()).select_from(self.model)

        # Add user filter if provided and model has created_by_id
        if user_id is not None and hasattr(self.model, "created_by_id"):
            query = query.where(self.model.created_by_id == user_id)

        filter_conditions = []
        for field, value in filters.items():
            if hasattr(self.model, field):
                filter_conditions.append(getattr(self.model, field) == value)

        if filter_conditions:
            query = query.where(and_(*filter_conditions))

        result = await self.session.execute(query)
        count = result.scalar_one_or_none() or 0
        return count > 0
