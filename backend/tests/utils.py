# tests/utils.py
"""
Utility functions for testing.
"""
import io
import json
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.orm.models import User, DatasetType, UserRole
from backend.app.services.auth import create_access_token


def get_test_file_path(filename: str) -> str:
    """Get the path to a test fixture file."""
    base_dir = Path(__file__).parent
    fixtures_dir = base_dir / "fixtures"
    return str(fixtures_dir / filename)


def read_test_file(filename: str) -> bytes:
    """Read a test fixture file's contents."""
    file_path = get_test_file_path(filename)
    with open(file_path, "rb") as f:
        return f.read()


def create_test_upload_file(
        filename: str = "test.csv",
        content: bytes = None,
        content_type: str = "text/csv"
) -> UploadFile:
    """Create a test UploadFile for file upload testing."""
    if content is None:
        content = b"header1,header2\nvalue1,value2\nvalue3,value4"

    return UploadFile(
        filename=filename,
        file=io.BytesIO(content),
        content_type=content_type
    )


def create_test_token(user: User, expires_minutes: int = 30) -> str:
    """Create a test JWT token for authentication."""
    return create_access_token(
        data={"sub": str(user.id), "email": user.email},
        expires_delta=expires_minutes
    )