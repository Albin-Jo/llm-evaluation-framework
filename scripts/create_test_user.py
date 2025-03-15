import asyncio
import sys
from uuid import uuid4

# Add the project root to Python path
sys.path.insert(0, ".")

from app.db.session import db_session
from app.models.orm.models import User, UserRole


async def create_test_user():
    async with db_session() as session:
        # Check if user already exists
        # Create a new test user
        test_user = User(
            id=uuid4(),
            external_id="test-external-id-123",
            email="test@example.com",
            display_name="Test User",
            role=UserRole.ADMIN,
            is_active=True
        )

        session.add(test_user)
        await session.commit()

        print(f"Created test user with ID: {test_user.id}")
        return test_user


if __name__ == "__main__":
    asyncio.run(create_test_user())