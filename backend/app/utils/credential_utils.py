import base64
import json
import logging
from typing import Dict, Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from backend.app.core.config import settings

# Configure logging
logger = logging.getLogger(__name__)


# Use a key derived from the app secret key
def _get_encryption_key():
    """
    Get or create encryption key derived from app secret.

    Returns:
        bytes: Encryption key
    """
    # Use app secret as base for key derivation
    app_secret = settings.APP_SECRET_KEY.get_secret_value().encode()

    # Static salt - in production, consider storing this securely
    salt = b'evaluation_framework_salt'

    # Key derivation function
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )

    key = base64.urlsafe_b64encode(kdf.derive(app_secret))
    return key


# Get encryption key
_fernet_key = _get_encryption_key()
_cipher = Fernet(_fernet_key)


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
        encrypted_data = _cipher.encrypt(json_data.encode())

        # Return base64 encoded string
        return base64.urlsafe_b64encode(encrypted_data).decode()
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
        # Decode from base64
        decoded_data = base64.urlsafe_b64decode(encrypted_data.encode())

        # Decrypt the data
        decrypted_data = _cipher.decrypt(decoded_data)

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
