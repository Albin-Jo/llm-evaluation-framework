import jwt
from datetime import datetime, timedelta, timezone
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import json
import uuid


def generate_test_token():
    # Generate a private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )

    # Get the public key in the format needed for JWKS
    public_key = private_key.public_key()
    public_numbers = public_key.public_numbers()

    # Create a key ID
    kid = str(uuid.uuid4())

    # Create JWKS
    jwks = {
        "keys": [
            {
                "kty": "RSA",
                "kid": kid,
                "use": "sig",
                "n": int_to_base64(public_numbers.n),
                "e": int_to_base64(public_numbers.e),
                "alg": "RS256"
            }
        ]
    }

    # Save JWKS to file (this will be served by our mock JWKS endpoint)
    with open("jwks.json", "w") as f:
        json.dump(jwks, f)

    print(f"JWKS file created: jwks.json")

    # Create payload with claims
    payload = {
        "sub": "test-external-id-123",  # Subject (user ID)
        "email": "test@example.com",
        "name": "Test User",
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(days=1),
        "aud": "your_client_id"  # Use your actual client ID from settings
    }

    # Get the private key in PEM format for JWT signing
    pem_private_key = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    # Create the JWT token with the kid in the header
    token = jwt.encode(
        payload,
        pem_private_key,
        algorithm="RS256",
        headers={"kid": kid}
    )

    print(f"Token: {token}")
    print(f"You can use this with the Authorization header: Bearer {token}")

    # Instructions for serving the JWKS
    print("\nTo use this token, you need to:")
    print("1. Serve the jwks.json file at a URL")
    print("2. Update your application settings to use this URL as the JWKS URI")
    print("3. Use the token in your API requests")

    return token, jwks


# Helper function to encode integers as base64url
def int_to_base64(value):
    import base64
    import struct

    # Convert the integer to bytes
    value_hex = format(value, 'x')
    # Ensure even length
    if len(value_hex) % 2 == 1:
        value_hex = '0' + value_hex
    value_bytes = bytes.fromhex(value_hex)

    # Encode to base64 and remove padding
    encoded = base64.urlsafe_b64encode(value_bytes).decode('ascii').rstrip('=')
    return encoded


if __name__ == "__main__":
    generate_test_token()