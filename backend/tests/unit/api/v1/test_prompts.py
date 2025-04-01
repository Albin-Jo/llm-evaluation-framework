import json
import uuid
import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.orm import Prompt
from backend.app.db.repositories.base import BaseRepository


class TestPromptsAPI:
    """Test suite for Prompt API endpoints."""

    @pytest.mark.asyncio
    async def test_create_prompt(
            self, async_client: httpx.AsyncClient, db: AsyncSession,
            test_user, user_auth_headers
    ):
        """Test prompt creation endpoint."""
        # Setup test data
        prompt_data = {
            "name": "Test Prompt",
            "description": "A test prompt",
            "content": "This is a prompt with {variable}",
            "variables": ["variable"],
            "is_public": False
        }

        # Test endpoint
        response = await async_client.post(
            "/api/prompts/",
            json=prompt_data,
            headers=user_auth_headers
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
            assert db_prompt.owner_id == test_user.id

    @pytest.mark.asyncio
    async def test_get_all_prompts(
            self, async_client: httpx.AsyncClient, db: AsyncSession,
            test_user, user_auth_headers
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
                variables=["var"],
                is_public=False,
                owner_id=test_user.id
            )
            for i in range(3)
        ]

        for prompt in prompts:
            db.add(prompt)
        await db.commit()

        # Test endpoint
        response = await async_client.get(
            "/api/prompts/",
            headers=user_auth_headers
        )

        # Assertions
        assert response.status_code == 200
        result = response.json()
        assert len(result) >= 3  # We should have at least our 3 prompts

        # Verify prompt properties in response
        prompt_names = [p["name"] for p in result]
        for i in range(3):
            assert f"Prompt {i}" in prompt_names

    @pytest.mark.asyncio
    async def test_get_prompt_by_id(
            self, async_client: httpx.AsyncClient, db: AsyncSession,
            test_user, user_auth_headers
    ):
        """Test endpoint for retrieving a specific prompt by ID."""
        # Create a test prompt
        prompt_id = uuid.uuid4()
        prompt = Prompt(
            id=prompt_id,
            name="Specific Prompt",
            description="Prompt for specific retrieval test",
            content="This is a specific prompt with {var}",
            variables=["var"],
            is_public=False,
            owner_id=test_user.id
        )
        db.add(prompt)
        await db.commit()

        # Test endpoint
        response = await async_client.get(
            f"/api/prompts/{prompt_id}",
            headers=user_auth_headers
        )

        # Assertions
        assert response.status_code == 200
        result = response.json()
        assert result["id"] == str(prompt_id)
        assert result["name"] == "Specific Prompt"
        assert result["owner_id"] == str(test_user.id)

    @pytest.mark.asyncio
    async def test_update_prompt(
            self, async_client: httpx.AsyncClient, db: AsyncSession,
            test_user, user_auth_headers
    ):
        """Test endpoint for updating a prompt."""
        # Create a test prompt
        prompt_id = uuid.uuid4()
        prompt = Prompt(
            id=prompt_id,
            name="Original Prompt",
            description="Original description",
            content="Original content with {var}",
            variables=["var"],
            is_public=False,
            owner_id=test_user.id
        )
        db.add(prompt)
        await db.commit()

        # Update data
        update_data = {
            "name": "Updated Prompt",
            "description": "Updated description",
            "content": "Updated content with {var} and {new_var}",
            "variables": ["var", "new_var"],
            "is_public": True
        }

        # Test endpoint
        response = await async_client.put(
            f"/api/prompts/{prompt_id}",
            json=update_data,
            headers=user_auth_headers
        )

        # Assertions
        assert response.status_code == 200
        result = response.json()
        assert result["name"] == "Updated Prompt"
        assert result["description"] == "Updated description"
        assert result["is_public"] is True
        assert "new_var" in result["variables"]

        # Verify database record was updated
        prompt_repo = BaseRepository(Prompt, db)
        updated_prompt = await prompt_repo.get(prompt_id)
        assert updated_prompt.name == "Updated Prompt"
        assert updated_prompt.description == "Updated description"
        assert updated_prompt.is_public is True
        assert "new_var" in updated_prompt.variables

    @pytest.mark.asyncio
    async def test_delete_prompt(
            self, async_client: httpx.AsyncClient, db: AsyncSession,
            test_user, user_auth_headers
    ):
        """Test endpoint for deleting a prompt."""
        # Create a test prompt
        prompt_id = uuid.uuid4()
        prompt = Prompt(
            id=prompt_id,
            name="Prompt to Delete",
            description="This prompt will be deleted",
            content="Content to delete with {var}",
            variables=["var"],
            is_public=False,
            owner_id=test_user.id
        )
        db.add(prompt)
        await db.commit()

        # Test endpoint
        response = await async_client.delete(
            f"/api/prompts/{prompt_id}",
            headers=user_auth_headers
        )

        # Assertions
        assert response.status_code == 204

        # Verify prompt was deleted from database
        prompt_repo = BaseRepository(Prompt, db)
        deleted_prompt = await prompt_repo.get(prompt_id)
        assert deleted_prompt is None

    @pytest.mark.asyncio
    async def test_render_prompt(
            self, async_client: httpx.AsyncClient, db: AsyncSession,
            test_user, user_auth_headers
    ):
        """Test endpoint for rendering a prompt with variables."""
        # Create a test prompt
        prompt_id = uuid.uuid4()
        prompt = Prompt(
            id=prompt_id,
            name="Render Test Prompt",
            description="Prompt for rendering test",
            content="Hello {name}, this is a {type} prompt.",
            variables=["name", "type"],
            is_public=False,
            owner_id=test_user.id
        )
        db.add(prompt)
        await db.commit()

        # Variables to render
        render_data = {
            "variables": {
                "name": "User",
                "type": "test"
            }
        }

        # Test endpoint
        response = await async_client.post(
            f"/api/prompts/{prompt_id}/render",
            json=render_data,
            headers=user_auth_headers
        )

        # Assertions
        assert response.status_code == 200
        result = response.json()
        assert "rendered_content" in result
        assert result["rendered_content"] == "Hello User, this is a test prompt."

    @pytest.mark.asyncio
    async def test_unauthorized_access(self, async_client: httpx.AsyncClient):
        """Test that unauthorized access is properly handled."""
        # Try to access endpoint without authentication
        response = await async_client.get("/api/prompts/")

        # Assertions - should either be 401 Unauthorized or 403 Forbidden
        assert response.status_code in (401, 403)

    @pytest.mark.asyncio
    async def test_prompt_validation_error(
            self, async_client: httpx.AsyncClient,
            user_auth_headers
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
            headers=user_auth_headers
        )

        # Assertions
        assert response.status_code == 422  # Unprocessable Entity
        result = response.json()
        assert "detail" in result

    @pytest.mark.asyncio
    async def test_template_based_prompt(
            self, async_client: httpx.AsyncClient, db: AsyncSession,
            test_user, user_auth_headers, sample_prompt_template
    ):
        """Test creating a prompt based on a template."""
        # Create a prompt template in the database
        template_id = sample_prompt_template["id"]

        # Setup prompt data with template reference
        prompt_data = {
            "name": "Template-based Prompt",
            "description": "A prompt based on a template",
            "template_id": str(template_id),
            "is_public": False
        }

        # Test endpoint
        response = await async_client.post(
            "/api/prompts/from-template",
            json=prompt_data,
            headers=user_auth_headers
        )

        # Assertions
        assert response.status_code in (200, 201)
        result = response.json()
        assert result["name"] == "Template-based Prompt"
        assert result["template_id"] == str(template_id)
        assert "id" in result