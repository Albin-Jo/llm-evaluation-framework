import uuid
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from backend.app.db.models.orm import Dataset, DatasetType
from backend.app.db.repositories.base import BaseRepository


class TestBaseRepositoryQueries:
    """Test suite for query operations of the BaseRepository class."""

    @pytest.mark.asyncio
    async def test_exists(self, db: AsyncSession, test_user):
        """Test checking if a record exists."""
        # Setup
        dataset_id = uuid.uuid4()
        nonexistent_id = uuid.uuid4()
        dataset = Dataset(
            id=dataset_id,
            name="Exists Test Dataset",
            description="Dataset for exists test",
            type=DatasetType.USER_QUERY,
            file_path="test/path/exists_test_dataset.json",
            is_public=False,
            owner_id=test_user.id,
            schema={},
            version="1.0",
            row_count=10
        )
        db.add(dataset)
        await db.commit()

        # Execute
        repo = BaseRepository(Dataset, db)
        exists_result = await repo.exists(dataset_id)
        nonexistent_result = await repo.exists(nonexistent_id)

        # Assert
        assert exists_result is True
        assert nonexistent_result is False

    @pytest.mark.asyncio
    async def test_get_multi(self, db: AsyncSession, test_user):
        """Test retrieving multiple records by IDs."""
        # Setup
        dataset_ids = [uuid.uuid4() for _ in range(3)]
        datasets = [
            Dataset(
                id=dataset_ids[i],
                name=f"Multi Dataset {i}",
                description=f"Dataset for multi test {i}",
                type=DatasetType.USER_QUERY,
                file_path=f"test/path/multi_dataset_{i}.json",
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

        # Add a non-requested dataset
        extra_dataset = Dataset(
            id=uuid.uuid4(),
            name="Extra Dataset",
            description="Extra dataset not in request",
            type=DatasetType.USER_QUERY,
            file_path="test/path/extra_dataset.json",
            is_public=False,
            owner_id=test_user.id,
            schema={},
            version="1.0",
            row_count=10
        )
        db.add(extra_dataset)
        await db.commit()

        # Execute
        repo = BaseRepository(Dataset, db)
        multi_datasets = await repo.get_multi(dataset_ids)

        # Assert
        assert len(multi_datasets) == 3
        retrieved_ids = [str(d.id) for d in multi_datasets]
        for dataset_id in dataset_ids:
            assert str(dataset_id) in retrieved_ids

    @pytest.mark.asyncio
    async def test_get_by_filter(self, db: AsyncSession, test_user):
        """Test retrieving records by filter."""
        # Setup
        # Create some test datasets with different public flags
        public_datasets = [
            Dataset(
                id=uuid.uuid4(),
                name=f"Public Dataset {i}",
                description=f"Public dataset for filter test {i}",
                type=DatasetType.USER_QUERY,
                file_path=f"test/path/public_dataset_{i}.json",
                is_public=True,
                owner_id=test_user.id,
                schema={},
                version="1.0",
                row_count=10
            )
            for i in range(3)
        ]

        private_datasets = [
            Dataset(
                id=uuid.uuid4(),
                name=f"Private Dataset {i}",
                description=f"Private dataset for filter test {i}",
                type=DatasetType.USER_QUERY,
                file_path=f"test/path/private_dataset_{i}.json",
                is_public=False,
                owner_id=test_user.id,
                schema={},
                version="1.0",
                row_count=10
            )
            for i in range(2)
        ]

        for dataset in public_datasets + private_datasets:
            db.add(dataset)
        await db.commit()

        # Execute
        repo = BaseRepository(Dataset, db)
        public_results = await repo.get_by_filter(is_public=True)
        private_results = await repo.get_by_filter(is_public=False)

        # Assert
        assert len(public_results) >= 3
        assert len(private_results) >= 2

        # Check if all public results are indeed public
        for dataset in public_results:
            assert dataset.is_public is True

        # Check if all private results are indeed private
        for dataset in private_results:
            assert dataset.is_public is False

    @pytest.mark.asyncio
    async def test_filter_with_multiple_criteria(self, db: AsyncSession, test_user):
        """Test filtering with multiple criteria."""
        # Setup
        # Create datasets with different combinations of attributes
        datasets = [
            # HR domain, public
            Dataset(
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
            ),
            # HR domain, private
            Dataset(
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
            ),
            # Finance domain, public
            Dataset(
                id=uuid.uuid4(),
                name="Finance Public Dataset",
                description="Public Finance dataset",
                type=DatasetType.USER_QUERY,
                file_path="test/path/finance_public.json",
                is_public=True,
                owner_id=test_user.id,
                domain="finance",
                schema={},
                version="1.0",
                row_count=10
            ),
            # Finance domain, private
            Dataset(
                id=uuid.uuid4(),
                name="Finance Private Dataset",
                description="Private Finance dataset",
                type=DatasetType.QUESTION_ANSWER,
                file_path="test/path/finance_private.json",
                is_public=False,
                owner_id=test_user.id,
                domain="finance",
                schema={},
                version="1.0",
                row_count=10
            )
        ]

        for dataset in datasets:
            db.add(dataset)
        await db.commit()

        # Execute
        repo = BaseRepository(Dataset, db)

        # Filter by domain and public flag
        hr_public_results = await repo.get_by_filter(domain="hr", is_public=True)
        hr_private_results = await repo.get_by_filter(domain="hr", is_public=False)
        finance_public_results = await repo.get_by_filter(domain="finance", is_public=True)

        # Filter by domain and type
        hr_context_results = await repo.get_by_filter(domain="hr", type=DatasetType.CONTEXT)
        finance_qa_results = await repo.get_by_filter(domain="finance", type=DatasetType.QUESTION_ANSWER)

        # Assert
        assert len(hr_public_results) >= 1
        assert len(hr_private_results) >= 1
        assert len(finance_public_results) >= 1
        assert len(hr_context_results) >= 1
        assert len(finance_qa_results) >= 1

        # Check specific datasets
        assert any(d.name == "HR Public Dataset" for d in hr_public_results)
        assert any(d.name == "HR Private Dataset" for d in hr_private_results)
        assert any(d.name == "Finance Public Dataset" for d in finance_public_results)
        assert any(d.name == "HR Private Dataset" for d in hr_context_results)
        assert any(d.name == "Finance Private Dataset" for d in finance_qa_results)

        # Verify exclusions
        assert not any(d.name == "HR Private Dataset" for d in hr_public_results)
        assert not any(d.name == "Finance Public Dataset" for d in hr_public_results)
        assert not any(d.name == "Finance Private Dataset" for d in hr_private_results)

    @pytest.mark.asyncio
    async def test_get_first_by_filter(self, db: AsyncSession, test_user):
        """Test retrieving the first record that matches a filter."""
        # Setup
        datasets = [
            Dataset(
                id=uuid.uuid4(),
                name="First Test Dataset 1",
                description="Dataset for first test",
                type=DatasetType.USER_QUERY,
                file_path="test/path/first_test1.json",
                is_public=True,
                owner_id=test_user.id,
                schema={},
                version="1.0",
                row_count=10
            ),
            Dataset(
                id=uuid.uuid4(),
                name="First Test Dataset 2",
                description="Dataset for first test",
                type=DatasetType.USER_QUERY,
                file_path="test/path/first_test2.json",
                is_public=True,
                owner_id=test_user.id,
                schema={},
                version="1.0",
                row_count=20
            )
        ]

        for dataset in datasets:
            db.add(dataset)
        await db.commit()

        # Execute
        repo = BaseRepository(Dataset, db)
        first_dataset = await repo.get_first_by_filter(is_public=True)

        # Assert
        assert first_dataset is not None
        assert first_dataset.is_public is True

        # Non-existent criteria
        non_existent = await repo.get_first_by_filter(name="Non Existent Dataset")
        assert non_existent is None

    @pytest.mark.asyncio
    async def test_count_by_filter(self, db: AsyncSession, test_user):
        """Test counting records that match a filter."""
        # Setup
        # Create datasets with different types
        user_query_datasets = [
            Dataset(
                id=uuid.uuid4(),
                name=f"User Query Dataset {i}",
                description=f"Dataset for count test {i}",
                type=DatasetType.USER_QUERY,
                file_path=f"test/path/user_query_{i}.json",
                is_public=True,
                owner_id=test_user.id,
                schema={},
                version="1.0",
                row_count=10
            )
            for i in range(3)
        ]

        context_datasets = [
            Dataset(
                id=uuid.uuid4(),
                name=f"Context Dataset {i}",
                description=f"Dataset for count test {i}",
                type=DatasetType.CONTEXT,
                file_path=f"test/path/context_{i}.json",
                is_public=True,
                owner_id=test_user.id,
                schema={},
                version="1.0",
                row_count=10
            )
            for i in range(2)
        ]

        for dataset in user_query_datasets + context_datasets:
            db.add(dataset)
        await db.commit()

        # Execute
        repo = BaseRepository(Dataset, db)
        user_query_count = await repo.count_by_filter(type=DatasetType.USER_QUERY)
        context_count = await repo.count_by_filter(type=DatasetType.CONTEXT)
        all_count = await repo.count_by_filter(is_public=True)

        # Assert
        assert user_query_count >= 3
        assert context_count >= 2
        assert all_count >= 5  # All public datasets (user_query + context)