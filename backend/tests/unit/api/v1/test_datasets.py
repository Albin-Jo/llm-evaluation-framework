import io
import json
import uuid
import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.orm import Dataset, DatasetType
from backend.app.db.repositories.base import BaseRepository


class TestDatasetsAPI:
    """Test suite for Dataset API endpoints."""

    @pytest.mark.asyncio
    async def test_create_dataset(
            self, async_client: httpx.AsyncClient, mock_storage_service,
            db: AsyncSession, test_user
    ):
        """Test dataset creation endpoint."""
        # Setup test data
        data = {
            "name": "Test Dataset",
            "description": "Dataset for testing purposes",
            "type": "user_query",
            "is_public": "false",
        }

        # Create a test JSON file
        file_content = json.dumps([
            {"query": "What is LLM?", "answer": "Large Language Model"},
            {"query": "How to evaluate?", "answer": "Use metrics"}
        ])

        # Create the upload file
        files = {
            "file": ("test_data.json", io.BytesIO(file_content.encode()), "application/json")
        }

        # Test endpoint using AsyncClient
        response = await async_client.post(
            "/api/datasets/",
            data=data,
            files=files,
        )

        # Assertions
        assert response.status_code in (200, 201), f"Expected 200 or 201, got {response.status_code}"

        # If we got a successful response, verify the results
        if response.status_code in (200, 201):
            result = response.json()
            assert result["name"] == "Test Dataset"
            assert "id" in result

            # Verify database record using the same session
            dataset_repo = BaseRepository(Dataset, db)
            db_dataset = await dataset_repo.get(uuid.UUID(result["id"]))
            assert db_dataset is not None
            assert db_dataset.name == "Test Dataset"
            assert db_dataset.owner_id == test_user.id

    @pytest.mark.asyncio
    async def test_get_all_datasets(
            self, async_client: httpx.AsyncClient, db: AsyncSession,
            test_user, user_auth_headers
    ):
        """Test endpoint for retrieving all datasets."""
        # Create some test datasets in the database
        dataset_repo = BaseRepository(Dataset, db)
        datasets = [
            Dataset(
                id=uuid.uuid4(),
                name=f"Dataset {i}",
                description=f"Test dataset {i}",
                type=DatasetType.USER_QUERY,
                file_path=f"test/path/dataset_{i}.json",
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

        # Test endpoint
        response = await async_client.get(
            "/api/datasets/",
            headers=user_auth_headers
        )

        # Assertions
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 3  # We should have at least our 3 datasets

        # Verify dataset properties in response
        dataset_names = [d["name"] for d in result]
        for i in range(3):
            assert f"Dataset {i}" in dataset_names

    @pytest.mark.asyncio
    async def test_get_dataset_by_id(
            self, async_client: httpx.AsyncClient, db: AsyncSession,
            test_user, user_auth_headers
    ):
        """Test endpoint for retrieving a specific dataset by ID."""
        # Create a test dataset
        dataset_id = uuid.uuid4()
        dataset = Dataset(
            id=dataset_id,
            name="Specific Dataset",
            description="Dataset for specific retrieval test",
            type=DatasetType.USER_QUERY,
            file_path="test/path/specific_dataset.json",
            is_public=False,
            owner_id=test_user.id,
            schema={},
            version="1.0",
            row_count=10
        )
        db.add(dataset)
        await db.commit()

        # Test endpoint
        response = await async_client.get(
            f"/api/datasets/{dataset_id}",
            headers=user_auth_headers
        )

        # Assertions
        assert response.status_code == 200
        result = response.json()
        assert result["id"] == str(dataset_id)
        assert result["name"] == "Specific Dataset"
        assert result["owner_id"] == str(test_user.id)

    @pytest.mark.asyncio
    async def test_update_dataset(
            self, async_client: httpx.AsyncClient, db: AsyncSession,
            test_user, user_auth_headers
    ):
        """Test endpoint for updating a dataset."""
        # Create a test dataset
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

        # Test endpoint
        response = await async_client.put(
            f"/api/datasets/{dataset_id}",
            json=update_data,
            headers=user_auth_headers
        )

        # Assertions
        assert response.status_code == 200
        result = response.json()
        assert result["name"] == "Updated Dataset"
        assert result["description"] == "Updated description"
        assert result["is_public"] is True

        # Verify database record was updated
        dataset_repo = BaseRepository(Dataset, db)
        updated_dataset = await dataset_repo.get(dataset_id)
        assert updated_dataset.name == "Updated Dataset"
        assert updated_dataset.description == "Updated description"
        assert updated_dataset.is_public is True

    @pytest.mark.asyncio
    async def test_delete_dataset(
            self, async_client: httpx.AsyncClient, db: AsyncSession,
            test_user, user_auth_headers
    ):
        """Test endpoint for deleting a dataset."""
        # Create a test dataset
        dataset_id = uuid.uuid4()
        dataset = Dataset(
            id=dataset_id,
            name="Dataset to Delete",
            description="This dataset will be deleted",
            type=DatasetType.USER_QUERY,
            file_path="test/path/delete_dataset.json",
            is_public=False,
            owner_id=test_user.id,
            schema={},
            version="1.0",
            row_count=10
        )
        db.add(dataset)
        await db.commit()

        # Test endpoint
        response = await async_client.delete(
            f"/api/datasets/{dataset_id}",
            headers=user_auth_headers
        )

        # Assertions
        assert response.status_code == 204

        # Verify dataset was deleted from database
        dataset_repo = BaseRepository(Dataset, db)
        deleted_dataset = await dataset_repo.get(dataset_id)
        assert deleted_dataset is None

    @pytest.mark.asyncio
    async def test_preview_dataset(
            self, async_client: httpx.AsyncClient, db: AsyncSession,
            test_user, user_auth_headers, mock_storage_service
    ):
        """Test endpoint for previewing dataset content."""
        # Create a test dataset
        dataset_id = uuid.uuid4()
        dataset = Dataset(
            id=dataset_id,
            name="Preview Dataset",
            description="Dataset for preview test",
            type=DatasetType.USER_QUERY,
            file_path="test/path/preview_dataset.json",
            is_public=False,
            owner_id=test_user.id,
            schema={},
            version="1.0",
            row_count=10
        )
        db.add(dataset)
        await db.commit()

        # Test endpoint
        response = await async_client.get(
            f"/api/datasets/{dataset_id}/preview",
            headers=user_auth_headers
        )

        # Assertions
        assert response.status_code == 200
        result = response.json()
        assert isinstance(result, list)
        # Our mock storage service should return a specific JSON structure
        assert len(result) > 0
        assert "query" in result[0]

    @pytest.mark.asyncio
    async def test_unauthorized_access(self, async_client: httpx.AsyncClient):
        """Test that unauthorized access is properly handled."""
        # Try to access endpoint without authentication
        response = await async_client.get("/api/datasets/")

        # Assertions - should either be 401 Unauthorized or 403 Forbidden
        assert response.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_dataset_validation_error(
            self, async_client: httpx.AsyncClient,
            user_auth_headers
    ):
        """Test API validation for dataset creation."""
        # Setup invalid data - missing required fields
        data = {
            "description": "Invalid dataset"
            # Missing name and type
        }

        # Create a test file
        files = {
            "file": ("test_data.json", io.BytesIO(b"{}"), "application/json")
        }

        # Test endpoint
        response = await async_client.post(
            "/api/datasets/",
            data=data,
            files=files,
            headers=user_auth_headers
        )

        # Assertions
        assert response.status_code == 422  # Unprocessable Entity
        result = response.json()
        assert "detail" in result