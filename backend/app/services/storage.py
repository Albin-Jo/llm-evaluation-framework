# File: app/services/storage.py
import os
import uuid
import time
import hashlib
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import BinaryIO, List, Optional, AsyncGenerator

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
        dir_path = self._get_full_path(directory)
        os.makedirs(dir_path, exist_ok=True)

        # Generate unique filename
        file_ext = os.path.splitext(file.filename)[1] if file.filename else ""
        filename = f"{uuid.uuid4()}{file_ext}"

        # Create full path
        relative_path = os.path.join(directory, filename)
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


class S3StorageService(BaseStorageService):
    """Storage service using Amazon S3."""

    def __init__(
            self,
            bucket_name: str,
            aws_access_key_id: str,
            aws_secret_access_key: str,
            region_name: str
    ):
        """
        Initialize the storage service.

        Args:
            bucket_name: S3 bucket name
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            region_name: AWS region name
        """
        import boto3

        self.bucket_name = bucket_name
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )

    async def upload_file(self, file: UploadFile, directory: str) -> str:
        """
        Upload a file to S3.

        Args:
            file: File to upload
            directory: Directory to store the file in

        Returns:
            str: Path to the uploaded file
        """
        # Generate unique filename
        file_ext = os.path.splitext(file.filename)[1] if file.filename else ""
        filename = f"{uuid.uuid4()}{file_ext}"

        # Create S3 key
        s3_key = os.path.join(directory, filename) if directory else filename

        # Read file content in chunks and upload
        content = await file.read()
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=s3_key,
            Body=content
        )

        # Reset file position for potential reuse
        await file.seek(0)

        return s3_key

    async def read_file(self, file_path: str) -> str:
        """
        Read a file from S3.

        Args:
            file_path: Path to the file

        Returns:
            str: File contents
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=file_path
            )
            return response["Body"].read().decode("utf-8")
        except Exception as e:
            raise FileNotFoundError(f"File not found: {file_path} ({str(e)})")

    async def read_file_stream(self, file_path: str, chunk_size: int = 8192) -> AsyncGenerator[bytes, None]:
        """
        Read a file from S3 as a stream.

        Args:
            file_path: Path to the file
            chunk_size: Size of each chunk to read

        Yields:
            bytes: File content chunks
        """
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=file_path
            )

            # Stream the response body
            body = response["Body"]
            while True:
                chunk = body.read(chunk_size)
                if not chunk:
                    break
                yield chunk

        except Exception as e:
            raise FileNotFoundError(f"File not found: {file_path} ({str(e)})")

    async def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from S3.

        Args:
            file_path: Path to the file

        Returns:
            bool: True if file was deleted, False otherwise
        """
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=file_path
            )
            return True
        except Exception:
            return False

    async def list_files(self, directory: str) -> List[str]:
        """
        List files in a directory.

        Args:
            directory: Directory to list files from

        Returns:
            List[str]: List of file paths
        """
        try:
            paginator = self.s3_client.get_paginator("list_objects_v2")
            files = []

            for page in paginator.paginate(
                    Bucket=self.bucket_name,
                    Prefix=directory
            ):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        files.append(obj["Key"])

            return files
        except Exception:
            return []

    async def generate_presigned_url(self, file_path: str, expiration: int = 3600) -> str:
        """
        Generate a presigned URL for an S3 object.

        Args:
            file_path: Path to the file
            expiration: Time in seconds until the presigned URL expires

        Returns:
            str: Presigned URL
        """
        try:
            response = self.s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.bucket_name,
                    'Key': file_path
                },
                ExpiresIn=expiration
            )
            return response
        except Exception as e:
            logger.error(f"Error generating presigned URL: {str(e)}")
            return None

    async def get_secure_url(self, file_path: str, expiration: int = 3600) -> str:
        """
        Get a secure URL for accessing an S3 file.

        Args:
            file_path: Path to the file
            expiration: Time in seconds until the URL expires

        Returns:
            str: Secure URL
        """
        return await self.generate_presigned_url(file_path, expiration)

    async def get_file_size(self, file_path: str) -> int:
        """
        Get the size of a file in bytes.

        Args:
            file_path: Path to the file

        Returns:
            int: File size in bytes
        """
        try:
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=file_path
            )
            return response.get('ContentLength', 0)
        except Exception as e:
            raise FileNotFoundError(f"File not found: {file_path} ({str(e)})")

    async def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in S3.

        Args:
            file_path: Path to the file

        Returns:
            bool: True if file exists, False otherwise
        """
        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=file_path
            )
            return True
        except Exception:
            return False


class AzureStorageService(BaseStorageService):
    """Storage service using Azure Blob Storage."""

    def __init__(
            self,
            connection_string: str,
            container_name: str
    ):
        """
        Initialize the storage service.

        Args:
            connection_string: Azure storage connection string
            container_name: Azure storage container name
        """
        from azure.storage.blob import BlobServiceClient

        self.connection_string = connection_string
        self.container_name = container_name
        self.blob_service_client = BlobServiceClient.from_connection_string(
            connection_string
        )
        self.container_client = self.blob_service_client.get_container_client(
            container_name
        )

    async def upload_file(self, file: UploadFile, directory: str) -> str:
        """
        Upload a file to Azure Blob Storage.

        Args:
            file: File to upload
            directory: Directory to store the file in

        Returns:
            str: Path to the uploaded file
        """
        # Generate unique filename
        file_ext = os.path.splitext(file.filename)[1] if file.filename else ""
        filename = f"{uuid.uuid4()}{file_ext}"

        # Create blob name
        blob_name = os.path.join(directory, filename) if directory else filename

        # Upload file
        content = await file.read()
        blob_client = self.container_client.get_blob_client(blob_name)
        blob_client.upload_blob(content, overwrite=True)

        # Reset file position
        await file.seek(0)

        return blob_name

    async def read_file(self, file_path: str) -> str:
        """
        Read a file from Azure Blob Storage.

        Args:
            file_path: Path to the file

        Returns:
            str: File contents
        """
        try:
            blob_client = self.container_client.get_blob_client(file_path)
            download_stream = blob_client.download_blob()
            return download_stream.readall().decode("utf-8")
        except Exception as e:
            raise FileNotFoundError(f"File not found: {file_path} ({str(e)})")

    async def read_file_stream(self, file_path: str, chunk_size: int = 8192) -> AsyncGenerator[bytes, None]:
        """
        Read a file from Azure Blob Storage as a stream.

        Args:
            file_path: Path to the file
            chunk_size: Size of each chunk to read

        Yields:
            bytes: File content chunks
        """
        try:
            blob_client = self.container_client.get_blob_client(file_path)
            download_stream = blob_client.download_blob()

            # Stream the data in chunks
            data = download_stream.readall()
            for i in range(0, len(data), chunk_size):
                yield data[i:i + chunk_size]

        except Exception as e:
            raise FileNotFoundError(f"File not found: {file_path} ({str(e)})")

    async def delete_file(self, file_path: str) -> bool:
        """
        Delete a file from Azure Blob Storage.

        Args:
            file_path: Path to the file

        Returns:
            bool: True if file was deleted, False otherwise
        """
        try:
            blob_client = self.container_client.get_blob_client(file_path)
            blob_client.delete_blob()
            return True
        except Exception:
            return False

    async def list_files(self, directory: str) -> List[str]:
        """
        List files in a directory.

        Args:
            directory: Directory to list files from

        Returns:
            List[str]: List of file paths
        """
        try:
            files = []
            blobs = self.container_client.list_blobs(name_starts_with=directory)

            for blob in blobs:
                files.append(blob.name)

            return files
        except Exception:
            return []

    async def get_secure_url(self, file_path: str, expiration: int = 3600) -> str:
        """
        Get a secure URL for accessing a file in Azure Blob Storage.

        Args:
            file_path: Path to the file
            expiration: Time in seconds until the URL expires

        Returns:
            str: Secure URL
        """
        from datetime import datetime, timedelta
        from azure.storage.blob import generate_blob_sas, BlobSasPermissions

        try:
            blob_client = self.container_client.get_blob_client(file_path)

            # Generate SAS token
            sas_token = generate_blob_sas(
                account_name=blob_client.account_name,
                container_name=self.container_name,
                blob_name=file_path,
                account_key=self.blob_service_client.credential.account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(seconds=expiration)
            )

            # Create the URL with SAS token
            return f"{blob_client.url}?{sas_token}"

        except Exception as e:
            logger.error(f"Error generating secure URL for Azure Blob: {str(e)}")
            return None

    async def get_file_size(self, file_path: str) -> int:
        """
        Get the size of a file in bytes.

        Args:
            file_path: Path to the file

        Returns:
            int: File size in bytes
        """
        try:
            blob_client = self.container_client.get_blob_client(file_path)
            properties = blob_client.get_blob_properties()
            return properties.size
        except Exception as e:
            raise FileNotFoundError(f"File not found: {file_path} ({str(e)})")

    async def file_exists(self, file_path: str) -> bool:
        """
        Check if a file exists in Azure Blob Storage.

        Args:
            file_path: Path to the file

        Returns:
            bool: True if file exists, False otherwise
        """
        try:
            blob_client = self.container_client.get_blob_client(file_path)
            return blob_client.exists()
        except Exception:
            return False


@lru_cache()
def get_storage_service() -> BaseStorageService:
    """
    Get the appropriate storage service based on configuration.

    Returns:
        BaseStorageService: Storage service
    """
    storage_type = settings.STORAGE_TYPE.lower()

    if storage_type == "s3":
        return S3StorageService(
            bucket_name=settings.STORAGE_S3_BUCKET,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
    elif storage_type == "azure":
        return AzureStorageService(
            connection_string=settings.AZURE_STORAGE_CONNECTION_STRING,
            container_name=settings.AZURE_STORAGE_CONTAINER
        )
    else:
        # Default to local storage
        return LocalStorageService(settings.STORAGE_LOCAL_PATH)