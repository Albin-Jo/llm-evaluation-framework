import io
import json
import uuid
import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.orm import Dataset
from backend.app.db.repositories.base import BaseRepository


class TestDatasetWorkflow:
    """Integration test suite for Dataset workflows."""

    @pytest.mark.asyncio
    async def test_create_update_preview_delete_workflow(
            self, async_client: httpx.AsyncClient, db: AsyncSession,
            test_user, user_auth_headers, mock_storage_service
    ):
        """Test the complete workflow of creating, updating, previewing and deleting a dataset."""

        # STEP 1: Create a dataset
        # Setup test data
        data = {
            "name": "Workflow Test Dataset",
            "description": "Dataset for workflow testing",
            "type": "question_answer",
            "is_public": "false",
        }

        # Create a test JSON file
        file_content = json.dumps([
            {"query": "What is LLM?", "answer": "Large Language Model"},
            {"query": "How to evaluate?", "answer": "Use metrics"}
        ])

        files = {
            "file": ("workflow_test.json", io.BytesIO(file_content.encode()), "application/json")
        }

        # Create dataset
        response = await async_client.post(
            "/api/datasets/",
            data=data,
            files=files,
            headers=user_auth_headers
        )

        assert response.status_code in (200, 201)
        created_dataset = response.json()
        dataset_id = created_dataset["id"]
        assert created_dataset["name"] == "Workflow Test Dataset"

        # STEP 2: Get the dataset
        response = await async_client.get(
            f"/api/datasets/{dataset_id}",
            headers=user_auth_headers
        )

        assert response.status_code == 200
        retrieved_dataset = response.json()
        assert retrieved_dataset["id"] == dataset_id
        assert retrieved_dataset["name"] == "Workflow Test Dataset"

        # STEP 3: Update the dataset
        update_data = {
            "name": "Updated Workflow Dataset",
            "description": "Updated workflow description",
            "is_public": True
        }

        response = await async_client.put(
            f"/api/datasets/{dataset_id}",
            json=update_data,
            headers=user_auth_headers
        )

        assert response.status_code == 200
        updated_dataset = response.json()
        assert updated_dataset["name"] == "Updated Workflow Dataset"
        assert updated_dataset["description"] == "Updated workflow description"
        assert updated_dataset["is_public"] is True

        # STEP 4: Get a preview of the dataset
        response = await async_client.get(
            f"/api/datasets/{dataset_id}/preview",
            headers=user_auth_headers
        )

        assert response.status_code == 200
        preview_data = response.json()
        assert isinstance(preview_data, list)
        assert len(preview_data) > 0
        assert "query" in preview_data[0]

        # STEP 5: Delete the dataset
        response = await async_client.delete(
            f"/api/datasets/{dataset_id}",
            headers=user_auth_headers
        )

        assert response.status_code == 204

        # Verify the dataset was deleted
        response = await async_client.get(
            f"/api/datasets/{dataset_id}",
            headers=user_auth_headers
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_dataset_filtering_workflow(
            self, async_client: httpx.AsyncClient, db: AsyncSession,
            test_user, user_auth_headers
    ):
        """Test workflow for filtering datasets by different criteria."""

        # STEP 1: Create several datasets with different properties
        # Public dataset with specific domain
        public_domain_dataset = Dataset(
            id=uuid.uuid4(),
            name="Public Domain Dataset",
            description="Public dataset with domain",
            type="user_query",
            file_path="test/path/public_domain_dataset.json",
            is_public=True,
            owner_id=test_user.id,
            domain="hr",
            schema={},
            version="1.0",
            row_count=10
        )

        # Private dataset with same domain
        private_domain_dataset = Dataset(
            id=uuid.uuid4(),
            name="Private Domain Dataset",
            description="Private dataset with domain",
            type="user_query",
            file_path="test/path/private_domain_dataset.json",
            is_public=False,
            owner_id=test_user.id,
            domain="hr",
            schema={},
            version="1.0",
            row_count=10
        )

        # Public dataset with different domain
        public_other_domain_dataset = Dataset(
            id=uuid.uuid4(),
            name="Public Other Domain",
            description="Public dataset with different domain",
            type="context",
            file_path="test/path/public_other_domain.json",
            is_public=True,
            owner_id=test_user.id,
            domain="finance",
            schema={},
            version="1.0",
            row_count=10
        )

        # Add datasets to database
        for dataset in [public_domain_dataset, private_domain_dataset, public_other_domain_dataset]:
            db.add(dataset)
        await db.commit()

        # STEP 2: Filter by domain
        response = await async_client.get(
            "/api/datasets/?domain=hr",
            headers=user_auth_headers
        )

        assert response.status_code == 200
        hr_datasets = response.json()
        assert len(hr_datasets) >= 2
        hr_names = [d["name"] for d in hr_datasets]
        assert "Public Domain Dataset" in hr_names
        assert "Private Domain Dataset" in hr_names

        # STEP 3: Filter by public flag
        response = await async_client.get(
            "/api/datasets/?is_public=true",
            headers=user_auth_headers
        )

        assert response.status_code == 200
        public_datasets = response.json()
        public_names = [d["name"] for d in public_datasets]
        assert "Public Domain Dataset" in public_names
        assert "Public Other Domain" in public_names
        assert "Private Domain Dataset" not in public_names

        # STEP 4: Filter by domain and public flag
        response = await async_client.get(
            "/api/datasets/?domain=hr&is_public=true",
            headers=user_auth_headers
        )

        assert response.status_code == 200
        filtered_datasets = response.json()
        filtered_names = [d["name"] for d in filtered_datasets]
        assert "Public Domain Dataset" in filtered_names
        assert "Private Domain Dataset" not in filtered_names
        assert "Public Other Domain" not in filtered_names

        # STEP 5: Filter by dataset type
        response = await async_client.get(
            "/api/datasets/?type=context",
            headers=user_auth_headers
        )

        assert response.status_code == 200
        context_datasets = response.json()
        context_names = [d["name"] for d in context_datasets]
        assert "Public Other Domain" in context_names
        assert "Public Domain Dataset" not in context_names
        assert "Private Domain Dataset" not in context_names