import hashlib
import os
import time
import uuid
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import List, AsyncGenerator

from fastapi import UploadFile

from backend.app.core.config import settings


class BaseStorageService(ABC):
    """Base class for storage services."""

    @abstractmethod
    async def upload_file(self, file: UploadFile, directory: str) -> str:
        """
        Upload a file to storage.

        Args:
            file: File to upload
            directory: Directory to store the file in

        Returns:
            str: Path to the uploaded file
        """
        pass

    @abstractmethod
    async def read_file(self, file_path: str) -> str:
        """
        Read a file from storage.

        Args:
            file_path: Path to the file

        Returns:
            str: File contents
        """
        pass

    @abstractmethod
    async def read_file_stream(self, file_path: str, chunk_size: int = 8192) -> AsyncGenerator[bytes, None]:
        """
        Read a file from storage as a stream.

        Args:
            file_path: Path to the file
            chunk_size: Size of each chunk to read

        Yields:
            bytes: File content chunks
        """
        pass

    @abstractmethod
    async def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from storage.

        Args:
            file_path: Path to the file

        Returns:
            bool: True if file was deleted, False otherwise
        """
        pass

    @abstractmethod
    async def list_files(self, directory: str) -> List[str]:
        """
        List files in a directory.

        Args:
            directory: Directory to list files from

        Returns:
            List[str]: List of file paths
        """
        pass

    @abstractmethod
    async def get_secure_url(self, file_path: str, expiration: int = 3600) -> str:
        """
        Get a secure URL for accessing a file.

        Args:
            file_path: Path to the file
            expiration: Time in seconds until the URL expires

        Returns:
            str: Secure URL
        """
        pass

    @abstractmethod
    async def get_file_size(self, file_path: str) -> int:
        """
        Get the size of a file in bytes.

        Args:
            file_path: Path to the file

        Returns:
            int: File size in bytes
        """
        pass

    @abstractmethod
    async def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists.

        Args:
            file_path: Path to the file

        Returns:
            bool: True if file exists, False otherwise
        """
        pass


class LocalStorageService(BaseStorageService):
    """Storage service using local filesystem."""

    def __init__(self, base_path: str):
        """
        Initialize the storage service.

        Args:
            base_path: Base path for file storage
        """
        self.base_path = base_path

        # Create base directory if it doesn't exist
        os.makedirs(self.base_path, exist_ok=True)

    def _get_full_path(self, path: str) -> str:
        """
        Get full path to a file.

        Args:
            path: Relative path

        Returns:
            str: Full path
        """
        return os.path.join(self.base_path, path)

    async def upload_file(self, file: UploadFile, directory: str) -> str:
        """
        Upload a file to local storage.

        Args:
            file: File to upload
            directory: Directory to store the file in

        Returns:
            str: Path to the uploaded file
        """
        # Create directory if it doesn't exist
        # Convert any backslashes to forward slashes for consistency across platforms
        directory = directory.replace('\\', '/')
        dir_path = self._get_full_path(directory)
        os.makedirs(dir_path, exist_ok=True)

        # Generate unique filename
        file_ext = os.path.splitext(file.filename)[1] if file.filename else ""
        filename = f"{uuid.uuid4()}{file_ext}"

        # Create full path
        relative_path = os.path.join(directory, filename).replace('\\', '/')
        full_path = self._get_full_path(relative_path)

        # Write file in chunks
        with open(full_path, "wb") as f:
            chunk_size = 8192  # 8KB chunks
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)

        # Reset file position for potential reuse
        await file.seek(0)

        return relative_path

    async def read_file(self, file_path: str) -> str:
        """
        Read a file from local storage.

        Args:
            file_path: Path to the file

        Returns:
            str: File contents
        """
        full_path = self._get_full_path(file_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(full_path, "r") as f:
            return f.read()

    async def read_file_stream(self, file_path: str, chunk_size: int = 8192) -> AsyncGenerator[bytes, None]:
        """
        Read a file from local storage as a stream.

        Args:
            file_path: Path to the file
            chunk_size: Size of each chunk to read

        Yields:
            bytes: File content chunks
        """
        full_path = self._get_full_path(file_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(full_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    async def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from local storage.

        Args:
            file_path: Path to the file

        Returns:
            bool: True if file was deleted, False otherwise
        """
        full_path = self._get_full_path(file_path)

        if not os.path.exists(full_path):
            return False

        os.remove(full_path)
        return True

    async def list_files(self, directory: str) -> List[str]:
        """
        List files in a directory.

        Args:
            directory: Directory to list files from

        Returns:
            List[str]: List of file paths
        """
        dir_path = self._get_full_path(directory)

        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            return []

        files = []
        for file in os.listdir(dir_path):
            file_path = os.path.join(directory, file)
            if os.path.isfile(self._get_full_path(file_path)):
                files.append(file_path)

        return files

    async def get_secure_url(self, file_path: str, expiration: int = 3600) -> str:
        """
        Get a secure URL for accessing a local file.

        In a real application, this would involve token generation and a protected endpoint.
        For simplicity, we're just returning a local path with a dummy token.

        Args:
            file_path: Path to the file
            expiration: Time in seconds until the URL expires

        Returns:
            str: Secure URL
        """
        # Create a simple time-based token
        timestamp = int(time.time()) + expiration
        token = hashlib.sha256(f"{file_path}:{timestamp}:{settings.APP_SECRET_KEY}".encode()).hexdigest()

        return f"/api/datasets/files/{file_path}?token={token}&expires={timestamp}"

    async def get_file_size(self, file_path: str) -> int:
        """
        Get the size of a file in bytes.

        Args:
            file_path: Path to the file

        Returns:
            int: File size in bytes
        """
        full_path = self._get_full_path(file_path)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        return os.path.getsize(full_path)

    async def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists.

        Args:
            file_path: Path to the file

        Returns:
            bool: True if file exists, False otherwise
        """
        full_path = self._get_full_path(file_path)
        return os.path.exists(full_path) and os.path.isfile(full_path)


@lru_cache()
def get_storage_service() -> BaseStorageService:
    """
    Get the appropriate storage service based on configuration.

    Returns:
        BaseStorageService: Storage service
    """
    return LocalStorageService(settings.STORAGE_LOCAL_PATH)
