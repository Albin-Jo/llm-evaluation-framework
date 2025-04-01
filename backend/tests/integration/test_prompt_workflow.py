import uuid
import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.orm import Prompt
from backend.app.db.repositories.base import BaseRepository


class TestPromptWorkflow:
    """Integration test suite for Prompt workflows."""

    @pytest.mark.asyncio
    async def test_create_update_render_delete_workflow(
            self, async_client: httpx.AsyncClient, db: AsyncSession,
            test_user, user_auth_headers
    ):
        """Test the complete workflow of creating, updating, rendering and deleting a prompt."""

        # STEP 1: Create a prompt
        prompt_data = {
            "name": "Workflow Test Prompt",
            "description": "Prompt for workflow testing",
            "content": "This is a {test_type} prompt with {variable}",
            "variables": ["test_type", "variable"],
            "is_public": False
        }

        response = await async_client.post(
            "/api/prompts/",
            json=prompt_data,
            headers=user_auth_headers
        )

        assert response.status_code in (200, 201)
        created_prompt = response.json()
        prompt_id = created_prompt["id"]
        assert created_prompt["name"] == "Workflow Test Prompt"
        assert "test_type" in created_prompt["variables"]

        # STEP 2: Get the prompt
        response = await async_client.get(
            f"/api/prompts/{prompt_id}",
            headers=user_auth_headers
        )

        assert response.status_code == 200
        updated_prompt = response.json()
        assert updated_prompt["name"] == "Updated Workflow Prompt"
        assert updated_prompt["description"] == "Updated workflow description"
        assert "new_var" in updated_prompt["variables"]
        assert updated_prompt["is_public"] is True

        # STEP 4: Render the prompt
        render_data = {
            "variables": {
                "test_type": "integration",
                "variable": "example",
                "new_var": "value"
            }
        }

        response = await async_client.post(
            f"/api/prompts/{prompt_id}/render",
            json=render_data,
            headers=user_auth_headers
        )

        assert response.status_code == 200
        render_result = response.json()
        assert "rendered_content" in render_result
        assert render_result["rendered_content"] == "This is an integration prompt with example and value"

        # STEP 5: Delete the prompt
        response = await async_client.delete(
            f"/api/prompts/{prompt_id}",
            headers=user_auth_headers
        )

        assert response.status_code == 204

        # Verify the prompt was deleted
        response = await async_client.get(
            f"/api/prompts/{prompt_id}",
            headers=user_auth_headers
        )

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_template_based_prompt_workflow(
            self, async_client: httpx.AsyncClient, db: AsyncSession,
            test_user, user_auth_headers
    , prompt_id=None):
        """Test the workflow for template-based prompts."""

        # STEP 1: Create a prompt template
        template_data = {
            "name": "Template Prompt",
            "description": "Template for workflow testing",
            "content": "This is a template with {var1} and {var2}",
            "variables": ["var1", "var2"],
            "is_public": True
        }

        response = await async_client.post(
            "/api/prompts/",
            json=template_data,
            headers=user_auth_headers
        )

        assert response.status_code in (200, 201)
        template = response.json()
        template_id = template["id"]

        # STEP 2: Create a prompt based on the template
        derived_data = {
            "name": "Derived Prompt",
            "description": "Prompt derived from template",
            "template_id": template_id,
            "is_public": False
        }

        response = await async_client.post(
            "/api/prompts/from-template",
            json=derived_data,
            headers=user_auth_headers
        )

        assert response.status_code in (200, 201)
        derived_prompt = response.json()
        derived_id = derived_prompt["id"]
        assert derived_prompt["name"] == "Derived Prompt"
        assert derived_prompt["template_id"] == template_id

        # STEP 3: Verify the derived prompt has inherited content and variables
        response = await async_client.get(
            f"/api/prompts/{derived_id}",
            headers=user_auth_headers
        )

        assert response.status_code == 200
        retrieved_prompt = response.json()
        assert retrieved_prompt["content"] == template["content"]
        assert retrieved_prompt["variables"] == template["variables"]

        # STEP 4: Render the derived prompt
        render_data = {
            "variables": {
                "var1": "template-based",
                "var2": "workflow"
            }
        }

        response = await async_client.post(
            f"/api/prompts/{derived_id}/render",
            json=render_data,
            headers=user_auth_headers
        )

        assert response.status_code == 200
        render_result = response.json()
        assert render_result["rendered_content"] == "This is a template with template-based and workflow"

        # STEP 5: Update the template
        update_data = {
            "content": "Updated template with {var1}, {var2} and {var3}",
            "variables": ["var1", "var2", "var3"]
        }

        response = await async_client.put(
            f"/api/prompts/{template_id}",
            json=update_data,
            headers=user_auth_headers
        )

        assert response.status_code == 200

        # STEP 6: Check if the derived prompt reflects template changes (if that's how your app works)
        # This depends on your implementation - if derived prompts should inherit template changes
        response = await async_client.get(
            f"/api/prompts/{derived_id}",
            headers=user_auth_headers
        )

        assert response.status_code == 200
        updated_derived = response.json()

        # If your implementation updates derived prompts when templates change:
        # assert "var3" in updated_derived["variables"]
        # Otherwise, it would still have the original variables

        # STEP 7: Clean up - delete both prompts
        await async_client.delete(f"/api/prompts/{derived_id}", headers=user_auth_headers)
        await async_client.delete(f"/api/prompts/{template_id}", headers=user_auth_headers)

        retrieved_prompt = response.json()
        assert retrieved_prompt["id"] == prompt_id
        assert retrieved_prompt["name"] == "Workflow Test Prompt"

        # STEP 3: Update the prompt
        update_data = {
            "name": "Updated Workflow Prompt",
            "description": "Updated workflow description",
            "content": "This is an {test_type} prompt with {variable} and {new_var}",
            "variables": ["test_type", "variable", "new_var"],
            "is_public": True
        }

        response = await async_client.put(
            f"/api/prompts/{prompt_id}",
            json=update_data,
            headers=user_auth_headers
        )

        assert response.status_code == 200