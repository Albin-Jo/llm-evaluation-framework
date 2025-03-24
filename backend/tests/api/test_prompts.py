# File: tests/api/test_prompts.py
import pytest
from httpx import AsyncClient
from uuid import uuid4, UUID

from backend.app.main import app
from backend.app.db.models.orm.models import PromptTemplate, Prompt, User, UserRole
from backend.app.db.schema.prompt_schema import (
    PromptTemplateCreate, PromptCreate, PromptFromTemplateCreate
)

# Test data
test_template = {
    "name": "Test Template",
    "description": "A template for testing",
    "template": "This is a test template with {variable1} and {variable2}",
    "variables": {
        "variable1": {"type": "string", "description": "First variable"},
        "variable2": {"type": "string", "description": "Second variable"}
    },
    "is_public": False,
    "version": "1.0.0"
}

test_jinja_template = {
    "name": "Jinja Test Template",
    "description": "A template using Jinja2",
    "template": "Hello {{ name }}! {% if show_greeting %}Welcome to {{ service }}.{% endif %}",
    "variables": {
        "name": {"type": "string", "description": "Name of the user"},
        "show_greeting": {"type": "boolean", "description": "Whether to show greeting"},
        "service": {"type": "string", "description": "Service name"}
    },
    "is_public": False,
    "version": "1.0.0"
}

test_prompt = {
    "name": "Test Prompt",
    "description": "A prompt for testing",
    "content": "This is a test prompt with {variable1} and {variable2}",
    "parameters": {
        "variable1": "value1",
        "variable2": "value2"
    },
    "is_public": False,
    "version": "1.0.0"
}


@pytest.fixture
async def admin_user(db_session):
    """Create an admin user for testing."""
    user = User(
        external_id="admin-test",
        email="admin@example.com",
        display_name="Admin User",
        role=UserRole.ADMIN,
        is_active=True
    )
    db_session.add(user)
    await db_session.flush()
    return user


@pytest.fixture
async def regular_user(db_session):
    """Create a regular user for testing."""
    user = User(
        external_id="user-test",
        email="user@example.com",
        display_name="Regular User",
        role=UserRole.EVALUATOR,
        is_active=True
    )
    db_session.add(user)
    await db_session.flush()
    return user


@pytest.fixture
async def test_template_obj(db_session, admin_user):
    """Create a test template."""
    template = PromptTemplate(**test_template)
    db_session.add(template)
    await db_session.flush()
    return template


@pytest.fixture
async def test_prompt_obj(db_session, regular_user, test_template_obj):
    """Create a test prompt."""
    prompt_data = test_prompt.copy()
    prompt_data["owner_id"] = regular_user.id
    prompt_data["template_id"] = test_template_obj.id
    prompt = Prompt(**prompt_data)
    db_session.add(prompt)
    await db_session.flush()
    return prompt


# Template Tests

@pytest.mark.asyncio
async def test_create_template(async_client, admin_token):
    """Test creating a prompt template."""
    response = await async_client.post(
        "/api/prompts/templates/",
        json=test_template,
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == test_template["name"]
    assert data["template"] == test_template["template"]
    assert "id" in data


@pytest.mark.asyncio
async def test_get_template(async_client, admin_token, test_template_obj):
    """Test getting a prompt template."""
    response = await async_client.get(
        f"/api/prompts/templates/{test_template_obj.id}",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == test_template["name"]
    assert data["id"] == str(test_template_obj.id)


@pytest.mark.asyncio
async def test_update_template(async_client, admin_token, test_template_obj):
    """Test updating a prompt template."""
    update_data = {
        "name": "Updated Template Name",
        "description": "Updated description"
    }
    response = await async_client.put(
        f"/api/prompts/templates/{test_template_obj.id}",
        json=update_data,
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == update_data["name"]
    assert data["description"] == update_data["description"]
    # Ensure other fields weren't changed
    assert data["template"] == test_template["template"]


@pytest.mark.asyncio
async def test_delete_template(async_client, admin_token, test_template_obj):
    """Test deleting a prompt template."""
    response = await async_client.delete(
        f"/api/prompts/templates/{test_template_obj.id}",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 204

    # Verify template is gone
    response = await async_client.get(
        f"/api/prompts/templates/{test_template_obj.id}",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_templates(async_client, admin_token, test_template_obj):
    """Test listing prompt templates."""
    response = await async_client.get(
        "/api/prompts/templates/",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    template_ids = [template["id"] for template in data]
    assert str(test_template_obj.id) in template_ids


@pytest.mark.asyncio
async def test_get_template_variables(async_client, admin_token, test_template_obj):
    """Test extracting variables from a template."""
    response = await async_client.get(
        f"/api/prompts/templates/{test_template_obj.id}/variables",
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert response.status_code == 200
    variables = response.json()
    assert isinstance(variables, list)
    assert "variable1" in variables
    assert "variable2" in variables


# Prompt Tests

@pytest.mark.asyncio
async def test_create_prompt(async_client, user_token, test_template_obj):
    """Test creating a prompt."""
    prompt_data = test_prompt.copy()
    prompt_data["template_id"] = str(test_template_obj.id)

    response = await async_client.post(
        "/api/prompts/",
        json=prompt_data,
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == prompt_data["name"]
    assert data["content"] == prompt_data["content"]
    assert "id" in data


@pytest.mark.asyncio
async def test_create_prompt_from_template(async_client, user_token, test_template_obj):
    """Test creating a prompt from a template."""
    prompt_data = {
        "name": "Generated Prompt",
        "description": "A prompt generated from a template",
        "variables": {
            "variable1": "test value 1",
            "variable2": "test value 2"
        },
        "is_public": False,
        "use_jinja": False
    }

    response = await async_client.post(
        f"/api/prompts/templates/{test_template_obj.id}/create",
        json=prompt_data,
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == prompt_data["name"]
    assert "test value 1" in data["content"]
    assert "test value 2" in data["content"]
    assert data["template_id"] == str(test_template_obj.id)


@pytest.mark.asyncio
async def test_get_prompt(async_client, user_token, test_prompt_obj):
    """Test getting a prompt."""
    response = await async_client.get(
        f"/api/prompts/{test_prompt_obj.id}",
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == test_prompt["name"]
    assert data["id"] == str(test_prompt_obj.id)


@pytest.mark.asyncio
async def test_update_prompt(async_client, user_token, test_prompt_obj):
    """Test updating a prompt."""
    update_data = {
        "name": "Updated Prompt Name",
        "description": "Updated description"
    }
    response = await async_client.put(
        f"/api/prompts/{test_prompt_obj.id}",
        json=update_data,
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == update_data["name"]
    assert data["description"] == update_data["description"]
    # Ensure other fields weren't changed
    assert data["content"] == test_prompt["content"]


@pytest.mark.asyncio
async def test_delete_prompt(async_client, user_token, test_prompt_obj):
    """Test deleting a prompt."""
    response = await async_client.delete(
        f"/api/prompts/{test_prompt_obj.id}",
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 204

    # Verify prompt is gone
    response = await async_client.get(
        f"/api/prompts/{test_prompt_obj.id}",
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_list_prompts(async_client, user_token, test_prompt_obj):
    """Test listing prompts."""
    response = await async_client.get(
        "/api/prompts/",
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    prompt_ids = [prompt["id"] for prompt in data]
    assert str(test_prompt_obj.id) in prompt_ids


@pytest.mark.asyncio
async def test_render_prompt(async_client, user_token, test_prompt_obj):
    """Test rendering a prompt with variables."""
    render_data = {
        "variables": {
            "variable1": "rendered value 1",
            "variable2": "rendered value 2"
        },
        "use_jinja": False
    }

    response = await async_client.post(
        f"/api/prompts/{test_prompt_obj.id}/render",
        json=render_data,
        headers={"Authorization": f"Bearer {user_token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["original"] == test_prompt["content"]
    assert "rendered value 1" in data["rendered"]
    assert "rendered value 2" in data["rendered"]
    assert data["success"] is True


@pytest.mark.asyncio
async def test_jinja_template_rendering(async_client, admin_token):
    """Test creating and rendering with a Jinja2 template."""
    # First create a Jinja template
    template_response = await async_client.post(
        "/api/prompts/templates/",
        json=test_jinja_template,
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert template_response.status_code == 200
    template_data = template_response.json()
    template_id = template_data["id"]

    # Now create a prompt from the template
    prompt_data = {
        "name": "Jinja Prompt",
        "description": "A prompt using Jinja2 templating",
        "variables": {
            "name": "John",
            "show_greeting": True,
            "service": "LLM Evaluation"
        },
        "is_public": False,
        "use_jinja": True
    }

    prompt_response = await async_client.post(
        f"/api/prompts/templates/{template_id}/create",
        json=prompt_data,
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert prompt_response.status_code == 200
    prompt_data = prompt_response.json()
    prompt_id = prompt_data["id"]

    # Verify the content was correctly rendered with Jinja2
    assert "Hello John!" in prompt_data["content"]
    assert "Welcome to LLM Evaluation" in prompt_data["content"]

    # Test rendering with different variables
    render_data = {
        "variables": {
            "name": "Alice",
            "show_greeting": False,
            "service": "AI Platform"
        },
        "use_jinja": True
    }

    render_response = await async_client.post(
        f"/api/prompts/{prompt_id}/render",
        json=render_data,
        headers={"Authorization": f"Bearer {admin_token}"}
    )
    assert render_response.status_code == 200
    render_result = render_response.json()
    assert "Hello Alice!" in render_result["rendered"]
    assert "Welcome to AI Platform" not in render_result["rendered"]