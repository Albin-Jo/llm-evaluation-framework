import uuid

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.orm import Prompt
from backend.app.db.repositories.base import BaseRepository


class TestPromptsAPI:
    """Test suite for Prompt API endpoints."""

    @pytest.mark.asyncio(loop_scope="session")
    async def test_create_prompt(
            self, async_client: httpx.AsyncClient, db: AsyncSession,
    ):
        """Test prompt creation endpoint."""
        # Setup test data
        prompt_data = {
            "name": "Test Prompt",
            "description": "A test prompt",
            "content": "This is a prompt with {variable}",
            "is_public": False
        }

        # Test endpoint
        response = await async_client.post(
            "/api/prompts/",
            json=prompt_data,
        )

        # Assertions
        assert response.status_code in (200, 201), f"Expected 200 or 201, got {response.status_code}"

        if response.status_code in (200, 201):
            result = response.json()
            assert result["name"] == "Test Prompt"
            assert "id" in result

            # Verify database record
            prompt_repo = BaseRepository(Prompt, db)
            db_prompt = await prompt_repo.get(uuid.UUID(result["id"]))
            assert db_prompt is not None
            assert db_prompt.name == "Test Prompt"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_get_all_prompts(
            self, async_client: httpx.AsyncClient, db: AsyncSession,
    ):
        """Test endpoint for retrieving all prompts."""
        # Create some test prompts in the database
        prompt_repo = BaseRepository(Prompt, db)
        prompts = [
            Prompt(
                id=uuid.uuid4(),
                name=f"Prompt {i}",
                description=f"Test prompt {i}",
                content=f"This is prompt content {i} with {{var}}",
                is_public=False,
            )
            for i in range(3)
        ]

        for prompt in prompts:
            db.add(prompt)
        await db.commit()

        # Test endpoint
        response = await async_client.get(
            "/api/prompts/",
        )

        # Assertions
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 3  # We should have at least our 3 prompts

        # Verify prompt properties in response
        prompt_names = [p["name"] for p in result]
        for i in range(3):
            assert f"Prompt {i}" in prompt_names

    @pytest.mark.asyncio(loop_scope="session")
    async def test_get_prompt_by_id(
            self, async_client: httpx.AsyncClient, db: AsyncSession,

    ):
        """Test endpoint for retrieving a specific prompt by ID."""
        # Create a test prompt
        prompt_id = uuid.uuid4()
        prompt = Prompt(
            id=prompt_id,
            name="Specific Prompt",
            description="Prompt for specific retrieval test",
            content="This is a specific prompt with {var}",
            is_public=False,
        )
        db.add(prompt)
        await db.commit()

        # Test endpoint
        response = await async_client.get(
            f"/api/prompts/{prompt_id}",
        )

        # Assertions
        assert response.status_code == 200
        result = response.json()
        assert result["id"] == str(prompt_id)
        assert result["name"] == "Specific Prompt"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_prompt_validation_error(
            self, async_client: httpx.AsyncClient,
    ):
        """Test API validation for prompt creation."""
        # Setup invalid data - missing required fields
        prompt_data = {
            "description": "Invalid prompt"
            # Missing name and content
        }

        # Test endpoint
        response = await async_client.post(
            "/api/prompts/",
            json=prompt_data,
        )

        # Assertions
        assert response.status_code == 422  # Unprocessable Entity
        result = response.json()
        assert "detail" in result
