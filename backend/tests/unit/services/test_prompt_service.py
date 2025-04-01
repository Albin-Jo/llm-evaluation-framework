import uuid
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.services.prompt import PromptService
from backend.app.db.models.orm import Prompt
from backend.app.schemas.prompt import PromptCreate, PromptUpdate, PromptRender


class TestPromptService:
    """Test suite for the PromptService."""

    @pytest.mark.asyncio
    async def test_create_prompt(self, db: AsyncSession, test_user):
        """Test creating a prompt through the service."""
        # Setup
        prompt_service = PromptService(db)

        # Prompt data
        prompt_data = PromptCreate(
            name="Service Test Prompt",
            description="Prompt for service test",
            content="This is a test prompt with {variable}",
            variables=["variable"],
            is_public=False
        )

        # Execute
        new_prompt = await prompt_service.create_prompt(
            prompt_data=prompt_data,
            owner_id=test_user.id
        )

        # Assertions
        assert new_prompt.id is not None
        assert new_prompt.name == "Service Test Prompt"
        assert new_prompt.content == "This is a test prompt with {variable}"
        assert "variable" in new_prompt.variables
        assert new_prompt.owner_id == test_user.id

    @pytest.mark.asyncio
    async def test_get_prompt(self, db: AsyncSession, test_user):
        """Test retrieving a prompt through the service."""
        # Setup
        prompt_id = uuid.uuid4()
        prompt = Prompt(
            id=prompt_id,
            name="Get Service Test Prompt",
            description="Prompt for get service test",
            content="This is a prompt for get test with {var}",
            variables=["var"],
            is_public=False,
            owner_id=test_user.id
        )
        db.add(prompt)
        await db.commit()

        # Execute
        prompt_service = PromptService(db)
        retrieved_prompt = await prompt_service.get_prompt(prompt_id)

        # Assertions
        assert retrieved_prompt is not None
        assert retrieved_prompt.id == prompt_id
        assert retrieved_prompt.name == "Get Service Test Prompt"

    @pytest.mark.asyncio
    async def test_get_all_prompts(self, db: AsyncSession, test_user):
        """Test retrieving all prompts through the service."""
        # Setup
        # Create some test prompts
        prompts = [
            Prompt(
                id=uuid.uuid4(),
                name=f"Service GetAll Prompt {i}",
                description=f"Prompt for service getall test {i}",
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

        # Execute
        prompt_service = PromptService(db)
        all_prompts = await prompt_service.get_prompts()

        # Assertions
        assert len(all_prompts) >= 3  # We may have other prompts from other tests

        # Check if our test prompts are in the results
        prompt_names = [p.name for p in all_prompts]
        for i in range(3):
            assert f"Service GetAll Prompt {i}" in prompt_names

    @pytest.mark.asyncio
    async def test_update_prompt(self, db: AsyncSession, test_user):
        """Test updating a prompt through the service."""
        # Setup
        prompt_id = uuid.uuid4()
        prompt = Prompt(
            id=prompt_id,
            name="Original Service Prompt",
            description="Original service description",
            content="Original content with {var}",
            variables=["var"],
            is_public=False,
            owner_id=test_user.id
        )
        db.add(prompt)
        await db.commit()

        # Update data
        update_data = PromptUpdate(
            name="Updated Service Prompt",
            description="Updated service description",
            content="Updated content with {var} and {new_var}",
            variables=["var", "new_var"],
            is_public=True
        )

        # Execute
        prompt_service = PromptService(db)
        updated_prompt = await prompt_service.update_prompt(prompt_id, update_data)

        # Assertions
        assert updated_prompt.id == prompt_id
        assert updated_prompt.name == "Updated Service Prompt"
        assert updated_prompt.description == "Updated service description"
        assert "new_var" in updated_prompt.variables
        assert updated_prompt.is_public is True

    @pytest.mark.asyncio
    async def test_delete_prompt(self, db: AsyncSession, test_user):
        """Test deleting a prompt through the service."""
        # Setup
        prompt_id = uuid.uuid4()
        prompt = Prompt(
            id=prompt_id,
            name="Delete Service Test Prompt",
            description="Prompt for delete service test",
            content="Content to delete with {var}",
            variables=["var"],
            is_public=False,
            owner_id=test_user.id
        )
        db.add(prompt)
        await db.commit()

        # Execute
        prompt_service = PromptService(db)
        success = await prompt_service.delete_prompt(prompt_id)

        # Assertions
        assert success is True

        # Verify prompt was deleted
        deleted = await prompt_service.get_prompt(prompt_id)
        assert deleted is None

    @pytest.mark.asyncio
    async def test_render_prompt(self, db: AsyncSession, test_user):
        """Test rendering a prompt with variables."""
        # Setup
        prompt_id = uuid.uuid4()
        prompt = Prompt(
            id=prompt_id,
            name="Render Service Test Prompt",
            description="Prompt for render service test",
            content="Hello {name}, this is a {type} prompt.",
            variables=["name", "type"],
            is_public=False,
            owner_id=test_user.id
        )
        db.add(prompt)
        await db.commit()

        # Render data
        render_data = PromptRender(
            variables={
                "name": "User",
                "type": "test"
            }
        )

        # Execute
        prompt_service = PromptService(db)
        rendered_content = await prompt_service.render_prompt(prompt_id, render_data)

        # Assertions
        assert rendered_content == "Hello User, this is a test prompt."

    @pytest.mark.asyncio
    async def test_get_user_prompts(self, db: AsyncSession, test_user):
        """Test retrieving prompts for a specific user through the service."""
        # Setup
        # Create prompts for our test user
        user_prompts = [
            Prompt(
                id=uuid.uuid4(),
                name=f"User Prompt {i}",
                description=f"User prompt test {i}",
                content=f"Content with {{var}} - {i}",
                variables=["var"],
                is_public=False,
                owner_id=test_user.id
            )
            for i in range(3)
        ]

        # Create a prompt for another user
        other_user_id = uuid.uuid4()
        other_prompt = Prompt(
            id=uuid.uuid4(),
            name="Other User Prompt",
            description="Prompt for another user",
            content="Other content with {var}",
            variables=["var"],
            is_public=False,
            owner_id=other_user_id
        )

        for prompt in user_prompts + [other_prompt]:
            db.add(prompt)
        await db.commit()

        # Execute
        prompt_service = PromptService(db)
        user_prompts_result = await prompt_service.get_user_prompts(test_user.id)

        # Assertions
        assert len(user_prompts_result) >= 3  # May have other prompts from other tests

        # Check that all returned prompts belong to our test user
        for prompt in user_prompts_result:
            assert prompt.owner_id == test_user.id

        # Check that no prompts from other users are returned
        other_user_prompts = [p for p in user_prompts_result if p.owner_id == other_user_id]
        assert len(other_user_prompts) == 0

    @pytest.mark.asyncio
    async def test_create_from_template(self, db: AsyncSession, test_user):
        """Test creating a prompt from a template."""
        # Setup
        # Create a template prompt
        template_id = uuid.uuid4()
        template = Prompt(
            id=template_id,
            name="Template Prompt",
            description="A template for other prompts",
            content="Template content with {var1} and {var2}",
            variables=["var1", "var2"],
            is_public=True,
            owner_id=test_user.id
        )
        db.add(template)
        await db.commit()

        # Execute
        prompt_service = PromptService(db)
        new_prompt = await prompt_service.create_from_template(
            name="Derived Prompt",
            description="Prompt derived from template",
            template_id=template_id,
            owner_id=test_user.id,
            is_public=False
        )

        # Assertions
        assert new_prompt.id is not None
        assert new_prompt.name == "Derived Prompt"
        assert new_prompt.template_id == template_id
        assert new_prompt.content == template.content
        assert set(new_prompt.variables) == set(template.variables)
        assert new_prompt.owner_id == test_user.id

    @pytest.mark.asyncio
    async def test_validate_prompt_access(self, db: AsyncSession, test_user):
        """Test validating prompt access."""
        # Setup
        # Create a prompt owned by test_user
        owned_prompt_id = uuid.uuid4()
        owned_prompt = Prompt(
            id=owned_prompt_id,
            name="Owned Prompt",
            description="Prompt owned by test_user",
            content="Content with {var}",
            variables=["var"],
            is_public=False,
            owner_id=test_user.id
        )

        # Create a prompt owned by another user
        another_user_id = uuid.uuid4()
        not_owned_prompt_id = uuid.uuid4()
        not_owned_prompt = Prompt(
            id=not_owned_prompt_id,
            name="Not Owned Prompt",
            description="Prompt not owned by test_user",
            content="Content with {var}",
            variables=["var"],
            is_public=False,
            owner_id=another_user_id
        )

        # Create a public prompt owned by another user
        public_prompt_id = uuid.uuid4()
        public_prompt = Prompt(
            id=public_prompt_id,
            name="Public Prompt",
            description="Public prompt not owned by test_user",
            content="Content with {var}",
            variables=["var"],
            is_public=True,
            owner_id=another_user_id
        )

        for prompt in [owned_prompt, not_owned_prompt, public_prompt]:
            db.add(prompt)
        await db.commit()

        # Execute
        prompt_service = PromptService(db)

        # Check validation results
        owned_access = await prompt_service.validate_prompt_access(owned_prompt_id, test_user.id)
        not_owned_access = await prompt_service.validate_prompt_access(not_owned_prompt_id, test_user.id)
        public_access = await prompt_service.validate_prompt_access(public_prompt_id, test_user.id)

        # Assertions
        assert owned_access is True  # User owns this prompt
        assert not_owned_access is False  # User doesn't own this private prompt
        assert public_access is True  # Prompt is public, so user has access

    @pytest.mark.asyncio
    async def test_search_prompts(self, db: AsyncSession, test_user):
        """Test searching prompts by query."""
        # Setup
        # Create prompts with specific content for search
        prompts = [
            Prompt(
                id=uuid.uuid4(),
                name="Machine Learning Evaluation",
                description="Prompt for evaluating ML models",
                content="Evaluate this machine learning model: {details}",
                variables=["details"],
                is_public=True,
                owner_id=test_user.id
            ),
            Prompt(
                id=uuid.uuid4(),
                name="Data Analysis",
                description="Prompt for data analysis",
                content="Analyze this dataset: {data}",
                variables=["data"],
                is_public=True,
                owner_id=test_user.id
            ),
            Prompt(
                id=uuid.uuid4(),
                name="Learning Rate Guide",
                description="Guide for setting learning rates in ML",
                content="Set learning rate based on these factors: {factors}",
                variables=["factors"],
                is_public=True,
                owner_id=test_user.id
            )
        ]

        for prompt in prompts:
            db.add(prompt)
        await db.commit()

        # Execute
        prompt_service = PromptService(db)
        ml_results = await prompt_service.search_prompts("machine learning")
        learning_results = await prompt_service.search_prompts("learning")
        data_results = await prompt_service.search_prompts("data")

        # Assertions
        assert len(ml_results) >= 1
        assert len(learning_results) >= 2
        assert len(data_results) >= 1

        # Check specific prompts in results
        ml_names = [p.name for p in ml_results]
        assert "Machine Learning Evaluation" in ml_names

        learning_names = [p.name for p in learning_results]
        assert "Machine Learning Evaluation" in learning_names
        assert "Learning Rate Guide" in learning_names

        data_names = [p.name for p in data_results]
        assert "Data Analysis" in data_names