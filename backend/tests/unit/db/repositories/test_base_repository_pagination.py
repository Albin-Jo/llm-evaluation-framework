import uuid
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from backend.app.db.models.orm import Dataset, Prompt, DatasetType
from backend.app.db.repositories.base import BaseRepository


class TestBaseRepositoryPagination:
    """Test suite for pagination and aggregation operations of the BaseRepository class."""

    @pytest.mark.asyncio
    async def test_paginate(self, db: AsyncSession, test_user):
        """Test paginating results."""
        # Setup - create 25 datasets
        datasets = [
            Dataset(
                id=uuid.uuid4(),
                name=f"Page Dataset {i}",
                description=f"Dataset for pagination test {i}",
                type=DatasetType.USER_QUERY,
                file_path=f"test/path/page_dataset_{i}.json",
                is_public=False,
                owner_id=test_user.id,
                schema={},
                version="1.0",
                row_count=10
            )
            for i in range(25)
        ]

        for dataset in datasets:
            db.add(dataset)
        await db.commit()

        # Execute
        repo = BaseRepository(Dataset, db)

        # Test first page (default page size is usually 10)
        page1 = await repo.paginate(page=1)
        # Test second page
        page2 = await repo.paginate(page=2)
        # Test with custom page size
        custom_page = await repo.paginate(page=1, page_size=5)

        # Assert
        assert len(page1.items) == 10
        assert len(page2.items) == 10
        assert len(custom_page.items) == 5

        # Test total count
        assert page1.total >= 25  # May have other datasets from other tests
        assert page1.total == page2.total

        # Test that pages don't contain the same items
        page1_ids = [str(item.id) for item in page1.items]
        page2_ids = [str(item.id) for item in page2.items]
        assert not any(id_val in page2_ids for id_val in page1_ids)

    @pytest.mark.asyncio
    async def test_paginate_with_filter(self, db: AsyncSession, test_user):
        """Test paginating filtered results."""
        # Setup - create datasets with different types
        user_query_datasets = [
            Dataset(
                id=uuid.uuid4(),
                name=f"User Query Dataset {i}",
                description=f"Dataset for filter pagination test {i}",
                type=DatasetType.USER_QUERY,
                file_path=f"test/path/paginate_user_query_{i}.json",
                is_public=True,
                owner_id=test_user.id,
                schema={},
                version="1.0",
                row_count=10
            )
            for i in range(15)
        ]

        context_datasets = [
            Dataset(
                id=uuid.uuid4(),
                name=f"Context Dataset {i}",
                description=f"Dataset for filter pagination test {i}",
                type=DatasetType.CONTEXT,
                file_path=f"test/path/paginate_context_{i}.json",
                is_public=True,
                owner_id=test_user.id,
                schema={},
                version="1.0",
                row_count=10
            )
            for i in range(10)
        ]

        for dataset in user_query_datasets + context_datasets:
            db.add(dataset)
        await db.commit()

        # Execute
        repo = BaseRepository(Dataset, db)

        # Paginate user query datasets
        user_query_page1 = await repo.paginate(
            page=1,
            page_size=5,
            type=DatasetType.USER_QUERY
        )

        user_query_page2 = await repo.paginate(
            page=2,
            page_size=5,
            type=DatasetType.USER_QUERY
        )

        # Paginate context datasets
        context_page = await repo.paginate(
            page=1,
            page_size=5,
            type=DatasetType.CONTEXT
        )

        # Assert
        assert len(user_query_page1.items) == 5
        assert len(user_query_page2.items) == 5
        assert len(context_page.items) == 5

        # Verify that we got the correct types
        for dataset in user_query_page1.items + user_query_page2.items:
            assert dataset.type == DatasetType.USER_QUERY

        for dataset in context_page.items:
            assert dataset.type == DatasetType.CONTEXT

        # Verify totals
        assert user_query_page1.total >= 15  # We should have at least our 15 user query datasets
        assert context_page.total >= 10  # We should have at least our 10 context datasets

    @pytest.mark.asyncio
    async def test_paginate_with_sorting(self, db: AsyncSession, test_user):
        """Test paginating with sorting."""
        # Setup - create datasets with specific names for sorting
        datasets = [
            Dataset(
                id=uuid.uuid4(),
                name=f"Sort Dataset {i:02d}",  # Add padding for consistent sorting
                description=f"Dataset for sorting test {i}",
                type=DatasetType.USER_QUERY,
                file_path=f"test/path/sort_dataset_{i}.json",
                is_public=True,
                owner_id=test_user.id,
                schema={},
                version="1.0",
                row_count=i * 10  # Different row counts for sorting
            )
            for i in range(1, 21)  # Create 20 datasets
        ]

        for dataset in datasets:
            db.add(dataset)
        await db.commit()

        # Execute
        repo = BaseRepository(Dataset, db)

        # Sort by name ascending
        name_asc_page = await repo.paginate(
            page=1,
            page_size=10,
            sort_by="name",
            sort_desc=False,
            name__like="Sort Dataset%"  # Filter to our test datasets
        )

        # Sort by name descending
        name_desc_page = await repo.paginate(
            page=1,
            page_size=10,
            sort_by="name",
            sort_desc=True,
            name__like="Sort Dataset%"  # Filter to our test datasets
        )

        # Sort by row_count ascending
        row_count_asc_page = await repo.paginate(
            page=1,
            page_size=10,
            sort_by="row_count",
            sort_desc=False,
            name__like="Sort Dataset%"  # Filter to our test datasets
        )

        # Sort by row_count descending
        row_count_desc_page = await repo.paginate(
            page=1,
            page_size=10,
            sort_by="row_count",
            sort_desc=True,
            name__like="Sort Dataset%"  # Filter to our test datasets
        )

        # Assert
        # Check name ascending sort
        name_asc_values = [item.name for item in name_asc_page.items]
        assert name_asc_values == sorted(name_asc_values)

        # Check name descending sort
        name_desc_values = [item.name for item in name_desc_page.items]
        assert name_desc_values == sorted(name_desc_values, reverse=True)

        # Check row_count ascending sort
        row_count_asc_values = [item.row_count for item in row_count_asc_page.items]
        assert row_count_asc_values == sorted(row_count_asc_values)

        # Check row_count descending sort
        row_count_desc_values = [item.row_count for item in row_count_desc_page.items]
        assert row_count_desc_values == sorted(row_count_desc_values, reverse=True)

    @pytest.mark.asyncio
    async def test_count(self, db: AsyncSession, test_user):
        """Test counting records."""
        # Setup
        # Create some test prompts
        prompts = [
            Prompt(
                id=uuid.uuid4(),
                name=f"Count Test Prompt {i}",
                description=f"Prompt for count test {i}",
                content=f"Test content {i} with {{var}}",
                variables=["var"],
                is_public=False,
                owner_id=test_user.id
            )
            for i in range(5)
        ]

        for prompt in prompts:
            db.add(prompt)
        await db.commit()

        # Execute
        repo = BaseRepository(Prompt, db)
        count = await repo.count()

        # Assert
        assert count >= 5  # We may have other prompts from other tests

        # Get actual count from database for verification
        stmt = select(func.count()).select_from(Prompt)
        result = await db.execute(stmt)
        db_count = result.scalar_one()
        assert count == db_count