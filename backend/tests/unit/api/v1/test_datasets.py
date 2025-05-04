import io
import json
import uuid
from typing import List

import httpx
import pytest
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.orm import Dataset, DatasetType
from backend.app.db.repositories.base import BaseRepository


async def _create_test_dataset(db: AsyncSession, **kwargs) -> Dataset:
    """Helper to create a test dataset with cleanup tracking."""
    dataset_id = kwargs.get("id", uuid.uuid4())
    dataset = Dataset(
        id=dataset_id,
        name=kwargs.get("name", "Test Dataset"),
        description=kwargs.get("description", "Dataset for testing purposes"),
        type=kwargs.get("type", DatasetType.USER_QUERY),
        file_path=kwargs.get("file_path", f"test/path/{dataset_id}.json"),
        is_public=kwargs.get("is_public", False),
        schema_definition=kwargs.get("schema_definition", {}),
        version=kwargs.get("version", "1.0"),
        row_count=kwargs.get("row_count", 10)
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    return dataset


async def _cleanup_datasets(db: AsyncSession, dataset_ids: List[uuid.UUID]):
    """Helper to clean up test datasets."""
    for dataset_id in dataset_ids:
        await db.execute(
            delete(Dataset).where(Dataset.id == dataset_id)
        )
    await db.commit()


class TestDatasetsAPI:
    """Test suite for Dataset API endpoints."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_create_dataset(
            self, async_client: httpx.AsyncClient, mock_storage_service,
            db: AsyncSession
    ):
        """Test dataset creation endpoint."""
        dataset_id = None
        try:
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
            result = response.json()
            assert result["name"] == "Test Dataset"
            assert "id" in result
            dataset_id = uuid.UUID(result["id"])

            # Verify database record using the same session
            dataset_repo = BaseRepository(Dataset, db)
            db_dataset = await dataset_repo.get(dataset_id)
            assert db_dataset is not None
            assert db_dataset.name == "Test Dataset"
        finally:
            # Clean up if we created a dataset
            if dataset_id:
                await _cleanup_datasets(db, [dataset_id])

    @pytest.mark.asyncio
    async def test_get_all_datasets(
            self, async_client: httpx.AsyncClient, db: AsyncSession
    ):
        """Test endpoint for retrieving all datasets."""
        created_dataset_ids = []  # Track created IDs for cleanup

        try:
            # Create test datasets
            datasets = [
                await _create_test_dataset(
                    db,
                    name=f"Dataset {i}",
                    description=f"Test dataset {i}"
                )
                for i in range(3)
            ]
            created_dataset_ids = [ds.id for ds in datasets]

            # Verify data was added
            result = await db.execute(select(Dataset).where(Dataset.id.in_(created_dataset_ids)))
            db_datasets = result.scalars().all()
            assert len(db_datasets) == 3, f"Expected 3 datasets, found {len(db_datasets)}"

            # Test endpoint
            response = await async_client.get(
                "/api/datasets/?skip=0&limit=100",
            )

            # Assertions
            assert response.status_code == 200
            result = response.json()
            assert len(result) >= 3, f"Expected at least 3 datasets, got {len(result)}"

            # Verify dataset properties in response
            dataset_names = [d["name"] for d in result]
            for i in range(3):
                assert f"Dataset {i}" in dataset_names, f"Dataset {i} not found in response"

        finally:
            # Clean up - remove the datasets we created
            await _cleanup_datasets(db, created_dataset_ids)

    @pytest.mark.asyncio(loop_scope="session")
    async def test_get_dataset_by_id(
            self, async_client: httpx.AsyncClient, db: AsyncSession,
    ):
        """Test endpoint for retrieving a specific dataset by ID."""
        dataset_id = None
        try:
            # Create a test dataset
            dataset = await _create_test_dataset(
                db,
                name="Specific Dataset",
                description="Dataset for specific retrieval test",
                is_public=True
            )
            dataset_id = dataset.id

            # Test endpoint
            response = await async_client.get(
                f"/api/datasets/{dataset_id}",
            )

            # Assertions
            assert response.status_code == 200
            result = response.json()
            assert result["id"] == str(dataset_id)
            assert result["name"] == "Specific Dataset"
        finally:
            if dataset_id:
                await _cleanup_datasets(db, [dataset_id])

    @pytest.mark.asyncio(loop_scope="session")
    async def test_update_dataset(
            self, async_client: httpx.AsyncClient, db: AsyncSession,
    ):
        """Test endpoint for updating a dataset."""
        dataset_id = None
        try:
            # Create a test dataset
            dataset = await _create_test_dataset(
                db,
                name="Original Dataset",
                description="Original description",
                is_public=False
            )
            dataset_id = dataset.id

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
        finally:
            if dataset_id:
                await _cleanup_datasets(db, [dataset_id])

    @pytest.mark.asyncio(loop_scope="session")
    async def test_delete_dataset(
            self, async_client: httpx.AsyncClient, db: AsyncSession
    ):
        """Test endpoint for deleting a dataset."""
        dataset_id = None
        try:
            # Create a test dataset
            dataset = await _create_test_dataset(
                db,
                name="Dataset to Delete",
                description="This dataset will be deleted"
            )
            dataset_id = dataset.id

            # Test endpoint
            response = await async_client.delete(
                f"/api/datasets/{dataset_id}",
            )

            # Assertions
            assert response.status_code == 204

            # Verify dataset was deleted from database
            dataset_repo = BaseRepository(Dataset, db)
            deleted_dataset = await dataset_repo.get(dataset_id)
            assert deleted_dataset is None

        finally:
            # In case delete failed in the API, clean up anyway
            if dataset_id:
                try:
                    dataset_repo = BaseRepository(Dataset, db)
                    if await dataset_repo.get(dataset_id):
                        await _cleanup_datasets(db, [dataset_id])
                except:
                    pass  # If cleanup fails, don't fail the test

    @pytest.mark.asyncio(loop_scope="session")
    async def test_dataset_validation_error(
            self, async_client: httpx.AsyncClient,
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
        )

        # Assertions
        assert response.status_code == 422
        result = response.json()
        assert "detail" in result

    @pytest.mark.asyncio(loop_scope="session")
    async def test_get_nonexistent_dataset(
            self, async_client: httpx.AsyncClient
    ):
        """Test retrieving a dataset that doesn't exist."""
        non_existent_id = uuid.uuid4()
        response = await async_client.get(
            f"/api/datasets/{non_existent_id}",
        )
        assert response.status_code == 404

    @pytest.mark.asyncio(loop_scope="session")
    async def test_dataset_pagination(
            self, async_client: httpx.AsyncClient, db: AsyncSession
    ):
        """Test pagination for dataset retrieval."""
        created_dataset_ids = []
        try:
            # Create 10 test datasets
            for i in range(10):
                dataset = await _create_test_dataset(
                    db,
                    name=f"Pagination Dataset {i}",
                    description=f"For testing pagination {i}"
                )
                created_dataset_ids.append(dataset.id)

            # Test first page (limit 5)
            response = await async_client.get("/api/datasets/?skip=0&limit=5")
            assert response.status_code == 200
            first_page = response.json()
            assert len(first_page) == 5

            # Test second page (limit 5, skip 5)
            response = await async_client.get("/api/datasets/?skip=5&limit=5")
            assert response.status_code == 200
            second_page = response.json()
            assert len(second_page) == 5

            # Ensure first and second page datasets are different
            first_page_ids = {item["id"] for item in first_page}
            second_page_ids = {item["id"] for item in second_page}
            assert not first_page_ids.intersection(second_page_ids)

        finally:
            await _cleanup_datasets(db, created_dataset_ids)
