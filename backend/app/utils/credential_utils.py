"""
Credential utilities for securely handling authentication credentials.

This module provides functions for encrypting, decrypting, and masking credentials.
"""
import base64
import json
import logging
import os
from typing import Dict, Any, Optional

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


# Generate or load encryption key
def get_encryption_key():
    """
    Get or create encryption key for credential encryption.

    Returns:
        bytes: Encryption key
    """
    # Try to get from environment variable first
    env_key = os.getenv("CREDENTIAL_ENCRYPTION_KEY")
    if env_key:
        try:
            # Ensure it's valid base64
            return base64.urlsafe_b64decode(env_key.encode())
        except Exception as e:
            logger.error(f"Invalid encryption key from environment: {e}")

    # Use a static key as fallback - in production, use env variable!
    # This is NOT secure for production, just for development
    fallback_key = b'Jt8MKKG6RYS42ZmvYYNbDT0Na4NFNg5J8xb7Yc4NS8c='

    # If this is production, log a warning
    if os.getenv("APP_ENV") == "production":
        logger.warning(
            "Using fallback encryption key in production! "
            "Set CREDENTIAL_ENCRYPTION_KEY environment variable for security."
        )

    return fallback_key


# Initialize encryption
_fernet = Fernet(get_encryption_key())


def encrypt_credentials(credentials: Dict[str, Any]) -> str:
    """
    Encrypt credentials dictionary to secure string.

    Args:
        credentials: Dictionary of credentials

    Returns:
        Encrypted string

    Raises:
        Exception: If encryption fails
    """
    try:
        # Convert dict to JSON string
        json_data = json.dumps(credentials)

        # Encrypt the data
        encrypted_data = _fernet.encrypt(json_data.encode())

        # Return as string
        return encrypted_data.decode()
    except Exception as e:
        logger.error(f"Error encrypting credentials: {e}")
        raise


def decrypt_credentials(encrypted_data: str) -> Dict[str, Any]:
    """
    Decrypt credentials from secure string.

    Args:
        encrypted_data: Encrypted credentials string

    Returns:
        Decrypted credentials dictionary

    Raises:
        Exception: If decryption fails
    """
    try:
        # Decrypt the data
        decrypted_data = _fernet.decrypt(encrypted_data.encode())

        # Parse JSON to dictionary
        return json.loads(decrypted_data.decode())
    except Exception as e:
        logger.error(f"Error decrypting credentials: {e}")
        raise


def mask_credentials(credentials: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a masked version of credentials for logging/display.

    Args:
        credentials: Original credentials

    Returns:
        Masked credentials
    """
    if not credentials:
        return {}

    masked = {}
    for key, value in credentials.items():
        if isinstance(value, str):
            # Mask all but first and last 2 chars
            if len(value) > 6:
                masked[key] = value[:2] + '*' * (len(value) - 4) + value[-2:]
            else:
                masked[key] = '******'
        else:
            masked[key] = '******'
    return masked