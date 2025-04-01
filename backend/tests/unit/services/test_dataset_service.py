import io
import json
import uuid
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.services.dataset import DatasetService
from backend.app.db.models.orm import Dataset, DatasetType
from backend.app.schemas.dataset import DatasetCreate, DatasetUpdate


class TestDatasetService:
    """Test suite for the DatasetService."""

    @pytest.mark.asyncio
    async def test_create_dataset(self, db: AsyncSession, test_user, mock_storage_service):
        """Test creating a dataset through the service."""
        # Setup
        dataset_service = DatasetService(db)

        # Create test file content
        file_content = json.dumps([
            {"query": "What is LLM?", "answer": "Large Language Model"},
            {"query": "How to evaluate?", "answer": "Use metrics"}
        ])
        test_file = MagicMock()
        test_file.filename = "test_data.json"
        test_file.file = io.BytesIO(file_content.encode())
        test_file.content_type = "application/json"

        # Dataset data
        dataset_data = DatasetCreate(
            name="Service Test Dataset",
            description="Dataset for service test",
            type=DatasetType.QUESTION_ANSWER,
            is_public=False
        )

        # Execute
        new_dataset = await dataset_service.create_dataset(
            dataset_data=dataset_data,
            file=test_file,
            owner_id=test_user.id
        )

        # Assertions
        assert new_dataset.id is not None
        assert new_dataset.name == "Service Test Dataset"
        assert new_dataset.type == DatasetType.QUESTION_ANSWER
        assert new_dataset.owner_id == test_user.id
        assert new_dataset.row_count > 0  # Should have calculated row count from JSON

    @pytest.mark.asyncio
    async def test_get_dataset(self, db: AsyncSession, test_user):
        """Test retrieving a dataset through the service."""
        # Setup
        dataset_id = uuid.uuid4()
        dataset = Dataset(
            id=dataset_id,
            name="Get Service Test Dataset",
            description="Dataset for get service test",
            type=DatasetType.USER_QUERY,
            file_path="test/path/get_service_test_dataset.json",
            is_public=False,
            owner_id=test_user.id,
            schema={},
            version="1.0",
            row_count=10
        )
        db.add(dataset)
        await db.commit()

        # Execute
        dataset_service = DatasetService(db)
        retrieved_dataset = await dataset_service.get_dataset(dataset_id)

        # Assertions
        assert retrieved_dataset is not None
        assert retrieved_dataset.id == dataset_id
        assert retrieved_dataset.name == "Get Service Test Dataset"

    @pytest.mark.asyncio
    async def test_get_all_datasets(self, db: AsyncSession, test_user):
        """Test retrieving all datasets through the service."""
        # Setup
        # Create some test datasets
        datasets = [
            Dataset(
                id=uuid.uuid4(),
                name=f"Service GetAll Dataset {i}",
                description=f"Dataset for service getall test {i}",
                type=DatasetType.USER_QUERY,
                file_path=f"test/path/service_getall_dataset_{i}.json",
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
        dataset_service = DatasetService(db)
        all_datasets = await dataset_service.get_datasets()

        # Assertions
        assert len(all_datasets) >= 3  # We may have other datasets from other tests

        # Check if our test datasets are in the results
        dataset_names = [d.name for d in all_datasets]
        for i in range(3):
            assert f"Service GetAll Dataset {i}" in dataset_names

    @pytest.mark.asyncio
    async def test_update_dataset(self, db: AsyncSession, test_user):
        """Test updating a dataset through the service."""
        # Setup
        dataset_id = uuid.uuid4()
        dataset = Dataset(
            id=dataset_id,
            name="Original Service Dataset",
            description="Original service description",
            type=DatasetType.USER_QUERY,
            file_path="test/path/original_service_dataset.json",
            is_public=False,
            owner_id=test_user.id,
            schema={},
            version="1.0",
            row_count=10
        )
        db.add(dataset)
        await db.commit()

        # Update data
        update_data = DatasetUpdate(
            name="Updated Service Dataset",
            description="Updated service description",
            is_public=True
        )

        # Execute
        dataset_service = DatasetService(db)
        updated_dataset = await dataset_service.update_dataset(dataset_id, update_data)

        # Assertions
        assert updated_dataset.id == dataset_id
        assert updated_dataset.name == "Updated Service Dataset"
        assert updated_dataset.description == "Updated service description"
        assert updated_dataset.is_public is True

    @pytest.mark.asyncio
    async def test_delete_dataset(self, db: AsyncSession, test_user, mock_storage_service):
        """Test deleting a dataset through the service."""
        # Setup
        dataset_id = uuid.uuid4()
        dataset = Dataset(
            id=dataset_id,
            name="Delete Service Test Dataset",
            description="Dataset for delete service test",
            type=DatasetType.USER_QUERY,
            file_path="test/path/delete_service_test_dataset.json",
            is_public=False,
            owner_id=test_user.id,
            schema={},
            version="1.0",
            row_count=10
        )
        db.add(dataset)
        await db.commit()

        # Execute
        dataset_service = DatasetService(db)
        success = await dataset_service.delete_dataset(dataset_id)

        # Assertions
        assert success is True

        # Verify dataset was deleted
        deleted = await dataset_service.get_dataset(dataset_id)
        assert deleted is None

    @pytest.mark.asyncio
    async def test_get_dataset_preview(self, db: AsyncSession, test_user, mock_storage_service):
        """Test retrieving a dataset preview through the service."""
        # Setup
        dataset_id = uuid.uuid4()
        dataset = Dataset(
            id=dataset_id,
            name="Preview Service Test Dataset",
            description="Dataset for preview service test",
            type=DatasetType.QUESTION_ANSWER,
            file_path="test/path/preview_service_test_dataset.json",
            is_public=False,
            owner_id=test_user.id,
            schema={},
            version="1.0",
            row_count=10
        )
        db.add(dataset)
        await db.commit()

        # Execute
        dataset_service = DatasetService(db)
        preview_data = await dataset_service.get_dataset_preview(dataset_id)

        # Assertions
        assert preview_data is not None
        assert isinstance(preview_data, list)
        # Our mock storage service should return specific JSON structure
        assert "query" in preview_data[0]

    @pytest.mark.asyncio
    async def test_get_user_datasets(self, db: AsyncSession, test_user):
        """Test retrieving datasets for a specific user through the service."""
        # Setup
        # Create datasets for our test user
        user_datasets = [
            Dataset(
                id=uuid.uuid4(),
                name=f"User Dataset {i}",
                description=f"User dataset test {i}",
                type=DatasetType.USER_QUERY,
                file_path=f"test/path/user_dataset_{i}.json",
                is_public=False,
                owner_id=test_user.id,
                schema={},
                version="1.0",
                row_count=10
            )
            for i in range(3)
        ]

        # Create a dataset for another user
        other_user_id = uuid.uuid4()
        other_dataset = Dataset(
            id=uuid.uuid4(),
            name="Other User Dataset",
            description="Dataset for another user",
            type=DatasetType.USER_QUERY,
            file_path="test/path/other_user_dataset.json",
            is_public=False,
            owner_id=other_user_id,
            schema={},
            version="1.0",
            row_count=10
        )

        for dataset in user_datasets + [other_dataset]:
            db.add(dataset)
        await db.commit()

        # Execute
        dataset_service = DatasetService(db)
        user_datasets_result = await dataset_service.get_user_datasets(test_user.id)

        # Assertions
        assert len(user_datasets_result) >= 3  # May have other datasets from other tests

        # Check that all returned datasets belong to our test user
        for dataset in user_datasets_result:
            assert dataset.owner_id == test_user.id

        # Check that no datasets from other users are returned
        other_user_datasets = [d for d in user_datasets_result if d.owner_id == other_user_id]
        assert len(other_user_datasets) == 0

    @pytest.mark.asyncio
    async def test_parse_dataset_file_json(self, db: AsyncSession, test_user, mock_storage_service):
        """Test parsing a JSON dataset file."""
        # Setup
        dataset_service = DatasetService(db)

        # Create test JSON file content
        file_content = json.dumps([
            {"query": "Question 1?", "answer": "Answer 1"},
            {"query": "Question 2?", "answer": "Answer 2"},
            {"query": "Question 3?", "answer": "Answer 3"}
        ])
        test_file = MagicMock()
        test_file.filename = "test_json.json"
        test_file.file = io.BytesIO(file_content.encode())
        test_file.content_type = "application/json"

        # Execute
        file_path, row_count, schema = await dataset_service._parse_dataset_file(
            file=test_file,
            dataset_id=uuid.uuid4()
        )

        # Assertions
        assert file_path is not None
        assert file_path.endswith(".json")
        assert row_count == 3
        assert schema is not None

    @pytest.mark.asyncio
    async def test_parse_dataset_file_csv(self, db: AsyncSession, test_user, mock_storage_service):
        """Test parsing a CSV dataset file."""
        # Setup
        dataset_service = DatasetService(db)

        # Create test CSV file content
        file_content = "query,answer\nQuestion 1?,Answer 1\nQuestion 2?,Answer 2\nQuestion 3?,Answer 3"
        test_file = MagicMock()
        test_file.filename = "test_csv.csv"
        test_file.file = io.BytesIO(file_content.encode())
        test_file.content_type = "text/csv"

        # Execute
        file_path, row_count, schema = await dataset_service._parse_dataset_file(
            file=test_file,
            dataset_id=uuid.uuid4()
        )

        # Assertions
        assert file_path is not None
        assert file_path.endswith(".csv")
        assert row_count == 3
        assert schema is not None

    @pytest.mark.asyncio
    async def test_filter_datasets(self, db: AsyncSession, test_user):
        """Test filtering datasets by various criteria."""
        # Setup
        # Create datasets with different attributes for filtering
        hr_public_dataset = Dataset(
            id=uuid.uuid4(),
            name="HR Public Dataset",
            description="Public HR dataset",
            type=DatasetType.USER_QUERY,
            file_path="test/path/hr_public.json",
            is_public=True,
            owner_id=test_user.id,
            domain="hr",
            schema={},
            version="1.0",
            row_count=10
        )

        hr_private_dataset = Dataset(
            id=uuid.uuid4(),
            name="HR Private Dataset",
            description="Private HR dataset",
            type=DatasetType.CONTEXT,
            file_path="test/path/hr_private.json",
            is_public=False,
            owner_id=test_user.id,
            domain="hr",
            schema={},
            version="1.0",
            row_count=10
        )

        finance_dataset = Dataset(
            id=uuid.uuid4(),
            name="Finance Dataset",
            description="Finance dataset",
            type=DatasetType.QUESTION_ANSWER,
            file_path="test/path/finance.json",
            is_public=True,
            owner_id=test_user.id,
            domain="finance",
            schema={},
            version="1.0",
            row_count=10
        )

        for dataset in [hr_public_dataset, hr_private_dataset, finance_dataset]:
            db.add(dataset)
        await db.commit()

        # Execute
        dataset_service = DatasetService(db)

        # Filter by domain
        hr_datasets = await dataset_service.get_datasets(domain="hr")
        # Filter by public flag
        public_datasets = await dataset_service.get_datasets(is_public=True)
        # Filter by type
        context_datasets = await dataset_service.get_datasets(type=DatasetType.CONTEXT)
        # Filter by multiple criteria
        hr_public_datasets = await dataset_service.get_datasets(domain="hr", is_public=True)

        # Assertions
        assert len(hr_datasets) >= 2
        assert len(public_datasets) >= 2
        assert len(context_datasets) >= 1
        assert len(hr_public_datasets) >= 1

        # Check specific datasets in results
        hr_dataset_names = [d.name for d in hr_datasets]
        assert "HR Public Dataset" in hr_dataset_names
        assert "HR Private Dataset" in hr_dataset_names

        public_dataset_names = [d.name for d in public_datasets]
        assert "HR Public Dataset" in public_dataset_names
        assert "Finance Dataset" in public_dataset_names

        context_dataset_names = [d.name for d in context_datasets]
        assert "HR Private Dataset" in context_dataset_names

        hr_public_dataset_names = [d.name for d in hr_public_datasets]
        assert "HR Public Dataset" in hr_public_dataset_names
        assert "HR Private Dataset" not in hr_public_dataset_names

    @pytest.mark.asyncio
    async def test_validate_dataset_ownership(self, db: AsyncSession, test_user):
        """Test validating dataset ownership."""
        # Setup
        # Create a dataset owned by test_user
        owned_dataset_id = uuid.uuid4()
        owned_dataset = Dataset(
            id=owned_dataset_id,
            name="Owned Dataset",
            description="Dataset owned by test_user",
            type=DatasetType.USER_QUERY,
            file_path="test/path/owned_dataset.json",
            is_public=False,
            owner_id=test_user.id,
            schema={},
            version="1.0",
            row_count=10
        )

        # Create a dataset owned by another user
        another_user_id = uuid.uuid4()
        not_owned_dataset_id = uuid.uuid4()
        not_owned_dataset = Dataset(
            id=not_owned_dataset_id,
            name="Not Owned Dataset",
            description="Dataset not owned by test_user",
            type=DatasetType.USER_QUERY,
            file_path="test/path/not_owned_dataset.json",
            is_public=False,
            owner_id=another_user_id,
            schema={},
            version="1.0",
            row_count=10
        )

        # Create a public dataset owned by another user
        public_dataset_id = uuid.uuid4()
        public_dataset = Dataset(
            id=public_dataset_id,
            name="Public Dataset",
            description="Public dataset not owned by test_user",
            type=DatasetType.USER_QUERY,
            file_path="test/path/public_dataset.json",
            is_public=True,
            owner_id=another_user_id,
            schema={},
            version="1.0",
            row_count=10
        )

        for dataset in [owned_dataset, not_owned_dataset, public_dataset]:
            db.add(dataset)
        await db.commit()

        # Execute
        dataset_service = DatasetService(db)

        # Check validation results
        owned_access = await dataset_service.validate_dataset_access(owned_dataset_id, test_user.id)
        not_owned_access = await dataset_service.validate_dataset_access(not_owned_dataset_id, test_user.id)
        public_access = await dataset_service.validate_dataset_access(public_dataset_id, test_user.id)

        # Assertions
        assert owned_access is True  # User owns this dataset
        assert not_owned_access is False  # User doesn't own this private dataset
        assert public_access is True  # Dataset is public, so user has access


import io
import json
import uuid
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.services.dataset import DatasetService
from backend.app.db.models.orm import Dataset, DatasetType
from backend.app.schemas.dataset import DatasetCreate, DatasetUpdate


class TestDatasetService:
    """Test suite for the DatasetService."""

    @pytest.mark.asyncio
    async def test_create_dataset(self, db: AsyncSession, test_user, mock_storage_service):
        """Test creating a dataset through the service."""
        # Setup
        dataset_service = DatasetService(db)

        # Create test file content
        file_content = json.dumps([
            {"query": "What is LLM?", "answer": "Large Language Model"},
            {"query": "How to evaluate?", "answer": "Use metrics"}
        ])
        test_file = MagicMock()
        test_file.filename = "test_data.json"
        test_file.file = io.BytesIO(file_content.encode())
        test_file.content_type = "application/json"

        # Dataset data
        dataset_data = DatasetCreate(
            name="Service Test Dataset",
            description="Dataset for service test",
            type=DatasetType.QUESTION_ANSWER,
            is_public=False
        )

        # Execute
        new_dataset = await dataset_service.create_dataset(
            dataset_data=dataset_data,
            file=test_file,
            owner_id=test_user.id
        )

        # Assertions
        assert updated_dataset.id == dataset_id
        assert updated_dataset.name == "Updated Service Dataset"
        assert updated_dataset.description == "Updated service description"
        assert updated_dataset.is_public is True

    @pytest.mark.asyncio
    async def test_delete_dataset(self, db: AsyncSession, test_user, mock_storage_service):
        """Test deleting a dataset through the service."""
        # Setup
        dataset_id = uuid.uuid4()
        dataset = Dataset(
            id=dataset_id,
            name="Delete Service Test Dataset",
            description="Dataset for delete service test",
            type=DatasetType.USER_QUERY,
            file_path="test/path/delete_service_test_dataset.json",
            is_public=False,
            owner_id=test_user.id,
            schema={},
            version="1.0",
            row_count=10
        )
        db.add(dataset)
        await db.commit()

        # Execute
        dataset_service = DatasetService(db)
        success = await dataset_service.delete_dataset(dataset_id)

        # Assertions
        assert success is True

        # Verify dataset was deleted
        deleted = await dataset_service.get_dataset(dataset_id)
        assert deleted is None

    @pytest.mark.asyncio
    async def test_get_dataset_preview(self, db: AsyncSession, test_user, mock_storage_service):
        """Test retrieving a dataset preview through the service."""
        # Setup
        dataset_id = uuid.uuid4()
        dataset = Dataset(
            id=dataset_id,
            name="Preview Service Test Dataset",
            description="Dataset for preview service test",
            type=DatasetType.QUESTION_ANSWER,
            file_path="test/path/preview_service_test_dataset.json",
            is_public=False,
            owner_id=test_user.id,
            schema={},
            version="1.0",
            row_count=10
        )
        db.add(dataset)
        await db.commit()

        # Execute
        dataset_service = DatasetService(db)
        preview_data = await dataset_service.get_dataset_preview(dataset_id)

        # Assertions
        assert preview_data is not None
        assert isinstance(preview_data, list)
        # Our mock storage service should return specific JSON structure
        assert "query" in preview_data[0]

    @pytest.mark.asyncio
    async def test_get_user_datasets(self, db: AsyncSession, test_user):
        """Test retrieving datasets for a specific user through the service."""
        # Setup
        # Create datasets for our test user
        user_datasets = [
            Dataset(
                id=uuid.uuid4(),
                name=f"User Dataset {i}",
                description=f"User dataset test {i}",
                type=DatasetType.USER_QUERY,
                file_path=f"test/path/user_dataset_{i}.json",
                is_public=False,
                owner_id=test_user.id,
                schema={},
                version="1.0",
                row_count=10
            )
            for i in range(3)
        ]

        # Create a dataset for another user
        other_user_id = uuid.uuid4()
        other_dataset = Dataset(
            id=uuid.uuid4(),
            name="Other User Dataset",
            description="Dataset for another user",
            type=DatasetType.USER_QUERY,
            file_path="test/path/other_user_dataset.json",
            is_public=False,
            owner_id=other_user_id,
            schema={},
            version="1.0",
            row_count=10
        )

        for dataset in user_datasets + [other_dataset]:
            db.add(dataset)
        await db.commit()

        # Execute
        dataset_service = DatasetService(db)
        user_datasets_result = await dataset_service.get_user_datasets(test_user.id)

        # Assertions
        assert len(user_datasets_result) >= 3  # May have other datasets from other tests

        # Check that all returned datasets belong to our test user
        for dataset in user_datasets_result:
            assert dataset.owner_id == test_user.id

        # Check that no datasets from other users are returned
        other_user_datasets = [d for d in user_datasets_result if d.owner_id == other_user_id]
        assert len(other_user_datasets) == 0

        assert new_dataset.id is not None
        assert new_dataset.name == "Service Test Dataset"
        assert new_dataset.type == DatasetType.QUESTION_ANSWER
        assert new_dataset.owner_id == test_user.id
        assert new_dataset.row_count > 0  # Should have calculated row count from JSON

    @pytest.mark.asyncio
    async def test_get_dataset(self, db: AsyncSession, test_user):
        """Test retrieving a dataset through the service."""
        # Setup
        dataset_id = uuid.uuid4()
        dataset = Dataset(
            id=dataset_id,
            name="Get Service Test Dataset",
            description="Dataset for get service test",
            type=DatasetType.USER_QUERY,
            file_path="test/path/get_service_test_dataset.json",
            is_public=False,
            owner_id=test_user.id,
            schema={},
            version="1.0",
            row_count=10
        )
        db.add(dataset)
        await db.commit()

        # Execute
        dataset_service = DatasetService(db)
        retrieved_dataset = await dataset_service.get_dataset(dataset_id)

        # Assertions
        assert retrieved_dataset is not None
        assert retrieved_dataset.id == dataset_id
        assert retrieved_dataset.name == "Get Service Test Dataset"

    @pytest.mark.asyncio
    async def test_get_all_datasets(self, db: AsyncSession, test_user):
        """Test retrieving all datasets through the service."""
        # Setup
        # Create some test datasets
        datasets = [
            Dataset(
                id=uuid.uuid4(),
                name=f"Service GetAll Dataset {i}",
                description=f"Dataset for service getall test {i}",
                type=DatasetType.USER_QUERY,
                file_path=f"test/path/service_getall_dataset_{i}.json",
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
        dataset_service = DatasetService(db)
        all_datasets = await dataset_service.get_datasets()

        # Assertions
        assert len(all_datasets) >= 3  # We may have other datasets from other tests

        # Check if our test datasets are in the results
        dataset_names = [d.name for d in all_datasets]
        for i in range(3):
            assert f"Service GetAll Dataset {i}" in dataset_names

    @pytest.mark.asyncio
    async def test_update_dataset(self, db: AsyncSession, test_user):
        """Test updating a dataset through the service."""
        # Setup
        dataset_id = uuid.uuid4()
        dataset = Dataset(
            id=dataset_id,
            name="Original Service Dataset",
            description="Original service description",
            type=DatasetType.USER_QUERY,
            file_path="test/path/original_service_dataset.json",
            is_public=False,
            owner_id=test_user.id,
            schema={},
            version="1.0",
            row_count=10
        )
        db.add(dataset)
        await db.commit()

        # Update data
        update_data = DatasetUpdate(
            name="Updated Service Dataset",
            description="Updated service description",
            is_public=True
        )

        # Execute
        dataset_service = DatasetService(db)
        updated_dataset = await dataset_service.update_dataset(dataset_id, update_data)

        # Assertions