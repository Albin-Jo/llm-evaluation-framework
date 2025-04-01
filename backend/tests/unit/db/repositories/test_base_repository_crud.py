import uuid
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.app.db.models.orm import Dataset, DatasetType
from backend.app.db.repositories.base import BaseRepository


class TestBaseRepositoryCRUD:
    """Test suite for basic CRUD operations of the BaseRepository class."""

    @pytest.mark.asyncio
    async def test_create(self, db: AsyncSession, test_user):
        """Test creating a new record."""
        # Setup
        repo = BaseRepository(Dataset, db)
        dataset_data = {
            "name": "Test Dataset",
            "description": "Dataset for repository test",
            "type": DatasetType.USER_QUERY,
            "file_path": "test/path/repo_test_dataset.json",
            "is_public": False,
            "owner_id": test_user.id,
            "schema": {},
            "version": "1.0",
            "row_count": 10
        }

        # Execute
        new_dataset = await repo.create(dataset_data)

        # Assert
        assert new_dataset.id is not None
        assert new_dataset.name == "Test Dataset"
        assert new_dataset.owner_id == test_user.id

        # Verify in database
        stmt = select(Dataset).where(Dataset.id == new_dataset.id)
        result = await db.execute(stmt)
        db_dataset = result.scalar_one_or_none()
        assert db_dataset is not None
        assert db_dataset.name == "Test Dataset"

    @pytest.mark.asyncio
    async def test_get(self, db: AsyncSession, test_user):
        """Test retrieving a record by ID."""
        # Setup
        dataset_id = uuid.uuid4()
        dataset = Dataset(
            id=dataset_id,
            name="Get Test Dataset",
            description="Dataset for get test",
            type=DatasetType.USER_QUERY,
            file_path="test/path/get_test_dataset.json",
            is_public=False,
            owner_id=test_user.id,
            schema={},
            version="1.0",
            row_count=10
        )
        db.add(dataset)
        await db.commit()

        # Execute
        repo = BaseRepository(Dataset, db)
        retrieved_dataset = await repo.get(dataset_id)

        # Assert
        assert retrieved_dataset is not None
        assert retrieved_dataset.id == dataset_id
        assert retrieved_dataset.name == "Get Test Dataset"

    @pytest.mark.asyncio
    async def test_get_all(self, db: AsyncSession, test_user):
        """Test retrieving all records."""
        # Setup
        # Create some test datasets
        datasets = [
            Dataset(
                id=uuid.uuid4(),
                name=f"GetAll Dataset {i}",
                description=f"Dataset for getall test {i}",
                type=DatasetType.USER_QUERY,
                file_path=f"test/path/getall_dataset_{i}.json",
                is_public=False,
                owner_id=test_user.id,
                schema={},
                version="1.0",
                row_count=10
            )
            for i in range(3)
        ]

        for dataset in datasets:
            db.add(dataset)
        await db.commit()

        # Execute
        repo = BaseRepository(Dataset, db)
        all_datasets = await repo.get_all()

        # Assert
        assert len(all_datasets) >= 3  # We may have other datasets from other tests

        # Check if our test datasets are in the results
        dataset_names = [d.name for d in all_datasets]
        for i in range(3):
            assert f"GetAll Dataset {i}" in dataset_names

    @pytest.mark.asyncio
    async def test_update(self, db: AsyncSession, test_user):
        """Test updating a record."""
        # Setup
        dataset_id = uuid.uuid4()
        dataset = Dataset(
            id=dataset_id,
            name="Original Dataset",
            description="Original description",
            type=DatasetType.USER_QUERY,
            file_path="test/path/original_dataset.json",
            is_public=False,
            owner_id=test_user.id,
            schema={},
            version="1.0",
            row_count=10
        )
        db.add(dataset)
        await db.commit()

        # Update data
        update_data = {
            "name": "Updated Dataset",
            "description": "Updated description",
            "is_public": True
        }

        # Execute
        repo = BaseRepository(Dataset, db)
        updated_dataset = await repo.update(dataset_id, update_data)

        # Assert
        assert updated_dataset.id == dataset_id
        assert updated_dataset.name == "Updated Dataset"
        assert updated_dataset.description == "Updated description"
        assert updated_dataset.is_public is True

        # Verify in database
        stmt = select(Dataset).where(Dataset.id == dataset_id)
        result = await db.execute(stmt)
        db_dataset = result.scalar_one_or_none()
        assert db_dataset.name == "Updated Dataset"
        assert db_dataset.description == "Updated description"
        assert db_dataset.is_public is True

    @pytest.mark.asyncio
    async def test_delete(self, db: AsyncSession, test_user):
        """Test deleting a record."""
        # Setup
        dataset_id = uuid.uuid4()
        dataset = Dataset(
            id=dataset_id,
            name="Delete Test Dataset",
            description="Dataset for delete test",
            type=DatasetType.USER_QUERY,
            file_path="test/path/delete_test_dataset.json",
            is_public=False,
            owner_id=test_user.id,
            schema={},
            version="1.0",
            row_count=10
        )
        db.add(dataset)
        await db.commit()

        # Execute
        repo = BaseRepository(Dataset, db)
        deleted = await repo.delete(dataset_id)

        # Assert
        assert deleted is True

        # Verify in database
        stmt = select(Dataset).where(Dataset.id == dataset_id)
        result = await db.execute(stmt)
        db_dataset = result.scalar_one_or_none()
        assert db_dataset is None