import uuid
import pytest
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.orm import Dataset, DatasetType, Prompt, User


class TestDatasetModel:
    """Test suite for Dataset model."""

    @pytest.mark.asyncio
    async def test_dataset_creation(self, db: AsyncSession, test_user):
        """Test creating a Dataset instance."""
        # Create a dataset instance
        dataset_id = uuid.uuid4()
        dataset = Dataset(
            id=dataset_id,
            name="Test Dataset Model",
            description="Dataset for model test",
            type=DatasetType.USER_QUERY,
            file_path="test/path/model_test_dataset.json",
            is_public=False,
            owner_id=test_user.id,
            schema={"type": "object", "properties": {}},
            version="1.0",
            row_count=10,
            domain="test"
        )

        # Add to database
        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)

        # Assertions
        assert dataset.id == dataset_id
        assert dataset.name == "Test Dataset Model"
        assert dataset.type == DatasetType.USER_QUERY
        assert dataset.owner_id == test_user.id
        assert isinstance(dataset.created_at, datetime)

        # Test relationships (if implemented)
        if hasattr(dataset, "owner"):
            assert dataset.owner.id == test_user.id

    @pytest.mark.asyncio
    async def test_dataset_enum_values(self, db: AsyncSession, test_user):
        """Test that DatasetType enum values work correctly."""
        # Test creating datasets with different enum values
        dataset_types = [
            (DatasetType.USER_QUERY, "user_query"),
            (DatasetType.CONTEXT, "context"),
            (DatasetType.QUESTION_ANSWER, "question_answer"),
            (DatasetType.CONVERSATION, "conversation"),
            (DatasetType.CUSTOM, "custom")
        ]

        for enum_value, string_value in dataset_types:
            dataset = Dataset(
                id=uuid.uuid4(),
                name=f"Dataset {string_value}",
                description=f"Dataset with type {string_value}",
                type=enum_value,
                file_path=f"test/path/{string_value}_dataset.json",
                is_public=False,
                owner_id=test_user.id,
                schema={},
                version="1.0",
                row_count=10
            )

            db.add(dataset)

        await db.commit()

        # Query the database to verify all types were saved correctly
        for enum_value, string_value in dataset_types:
            dataset = await db.get(Dataset, {"name": f"Dataset {string_value}"})
            assert dataset is not None
            assert dataset.type == enum_value


class TestPromptModel:
    """Test suite for Prompt model."""

    @pytest.mark.asyncio
    async def test_prompt_creation(self, db: AsyncSession, test_user):
        """Test creating a Prompt instance."""
        # Create a prompt instance
        prompt_id = uuid.uuid4()
        prompt = Prompt(
            id=prompt_id,
            name="Test Prompt Model",
            description="Prompt for model test",
            content="This is a test prompt with {var1} and {var2}",
            variables=["var1", "var2"],
            is_public=True,
            owner_id=test_user.id
        )

        # Add to database
        db.add(prompt)
        await db.commit()
        await db.refresh(prompt)

        # Assertions
        assert prompt.id == prompt_id
        assert prompt.name == "Test Prompt Model"
        assert "var1" in prompt.variables
        assert "var2" in prompt.variables
        assert prompt.is_public is True
        assert prompt.owner_id == test_user.id
        assert isinstance(prompt.created_at, datetime)

        # Test relationships (if implemented)
        if hasattr(prompt, "owner"):
            assert prompt.owner.id == test_user.id

    @pytest.mark.asyncio
    async def test_prompt_template_relationship(self, db: AsyncSession, test_user):
        """Test the relationship between Prompt and its template."""
        # Create a template prompt
        template_id = uuid.uuid4()
        template = Prompt(
            id=template_id,
            name="Template Prompt",
            description="A template for other prompts",
            content="Template content with {var}",
            variables=["var"],
            is_public=True,
            owner_id=test_user.id
        )

        # Create a prompt based on the template
        derived_prompt = Prompt(
            id=uuid.uuid4(),
            name="Derived Prompt",
            description="Prompt based on a template",
            content="Derived content with {var}",
            variables=["var"],
            is_public=False,
            template_id=template_id,
            owner_id=test_user.id
        )

        # Add to database
        db.add(template)
        db.add(derived_prompt)
        await db.commit()
        await db.refresh(derived_prompt)

        # Assertions
        assert derived_prompt.template_id == template_id

        # Test bidirectional relationship if implemented
        if hasattr(derived_prompt, "template") and hasattr(template, "derived_prompts"):
            await db.refresh(template)
            assert derived_prompt.template.id == template_id
            assert derived_prompt.id in [p.id for p in template.derived_prompts]