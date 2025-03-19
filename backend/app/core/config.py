import os
import logging
from typing import List, Optional, Union, Dict, Any

from pydantic import model_validator, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""
    # App settings
    APP_NAME: str = "Microservice Pixi LLM Evaluation"
    APP_DESCRIPTION: str = "A framework for evaluating LLM-based applications"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = "development"  # Default to development for easier local setup
    APP_DEBUG: bool = True  # Default to True for development
    APP_SECRET_KEY: SecretStr = Field(default=SecretStr("dev_secret_key_change_me"))
    APP_BASE_URL: str = "http://localhost:8000"

    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]

    # Database settings
    DB_HOST: str = "localhost"
    DB_PORT: str = "5432"
    DB_USER: str = "postgres"
    DB_PASSWORD: SecretStr = Field(default=SecretStr("postgres"))
    DB_NAME: str = "pixi_eval"
    DB_URI: Optional[str] = None  # Will be computed if not provided

    # Celery settings
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # OpenAI settings
    OPENAI_API_KEY: SecretStr = Field(default=SecretStr("sk-dummy-key"))

    # OIDC settings
    OIDC_DISCOVERY_URL: str = "https://example.auth0.com/.well-known/openid-configuration"
    OIDC_CLIENT_ID: str = "your_client_id"
    OIDC_CLIENT_SECRET: SecretStr = Field(default=SecretStr("your_client_secret"))

    # Redis settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: str = "6379"

    STORAGE_TYPE: str = "local"
    STORAGE_LOCAL_PATH: str = "storage"

    # Convert to int during initialization to properly handle env vars
    MAX_UPLOAD_SIZE: int = 52428800  # Default: 50MB (50 * 1024 * 1024)

    # Additional model config
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),  # Try loading from multiple files
        env_file_encoding="utf-8",
        case_sensitive=True,  # Most env vars are uppercase by convention
        extra="ignore",  # Allow extra fields
        validate_default=True,  # Validate default values
    )

    @model_validator(mode='after')
    def check_production_settings(self) -> 'Settings':
        """Validate production configuration has appropriate settings."""
        if self.APP_ENV == "production":
            # Ensure CORS is properly restricted in production
            if "*" in self.CORS_ORIGINS:
                raise ValueError("Wildcard CORS origin '*' is not allowed in production")

            # Ensure debug is disabled in production
            if self.APP_DEBUG:
                raise ValueError("DEBUG mode should not be enabled in production")

            # Check for default development keys in production
            if self.APP_SECRET_KEY.get_secret_value() == "dev_secret_key_change_me":
                raise ValueError("Default APP_SECRET_KEY cannot be used in production")

            if self.DB_PASSWORD.get_secret_value() == "postgres":
                raise ValueError("Default database password cannot be used in production")

            if self.OPENAI_API_KEY.get_secret_value() == "sk-dummy-key":
                raise ValueError("Default OpenAI API key cannot be used in production")

        return self

    @model_validator(mode='after')
    def ensure_db_uri(self) -> 'Settings':
        """Ensure DB_URI is always set with a valid value."""
        # If DB_URI is explicitly set, use it
        if self.DB_URI:
            return self

        # For testing, use SQLite
        if self.APP_ENV == "testing":
            self.DB_URI = "sqlite+aiosqlite:///:memory:"
        else:
            # Extract password value - we need to handle it as SecretStr
            db_password = self.DB_PASSWORD.get_secret_value()

            # Build PostgreSQL connection string
            self.DB_URI = (
                f"postgresql+asyncpg://{self.DB_USER}:{db_password}"
                f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
            )

        # Log the DB_URI for debugging (excluding password)
        masked_uri = self.get_masked_db_uri()
        logger = logging.getLogger("settings")
        logger.debug(f"Using database URI: {masked_uri}")

        return self

    def get_masked_db_uri(self) -> str:
        """Return DB_URI with password masked for safe logging."""
        if not self.DB_URI:
            return "None"

        # Simple masking by replacing password portion
        parts = self.DB_URI.split('@')
        if len(parts) > 1:
            auth_parts = parts[0].split(':')
            if len(auth_parts) > 2:
                masked = f"{auth_parts[0]}:****@{parts[1]}"
                return masked

        # If parsing fails, return a fully masked version
        return "****MASKED-CONNECTION-STRING****"

    def __str__(self) -> str:
        """Create a readable string representation without sensitive data."""
        sensitive_fields = [
            "DB_PASSWORD", "APP_SECRET_KEY", "OPENAI_API_KEY",
            "OIDC_CLIENT_SECRET"
        ]

        output = ["Settings:"]
        for key, value in self.__dict__.items():
            if key == "DB_URI":
                output.append(f"  {key}: {self.get_masked_db_uri()}")
            elif key in sensitive_fields or isinstance(value, SecretStr):
                output.append(f"  {key}: ****HIDDEN****")
            else:
                output.append(f"  {key}: {value}")

        return "\n".join(output)


def get_settings() -> Settings:
    """
    Get settings singleton with appropriate error handling.

    Returns:
        Settings: Configured settings object
    """
    # Configure logging
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("settings")

    try:
        # Load settings
        settings = Settings()

        # Log settings in debug mode
        if settings.APP_DEBUG:
            logger.info(f"Loaded settings for environment: {settings.APP_ENV}")
            logger.debug(f"{settings}")

        return settings

    except Exception as e:
        # Provide helpful error information
        env = os.getenv("APP_ENV", "development")
        logger.error(f"Failed to load settings: {str(e)}")

        if env != "production":
            print(f"\n{'=' * 80}")
            print(f"ERROR LOADING SETTINGS: {str(e)}")
            print(f"Make sure to create a .env or .env.local file with required configuration")
            print(f"{'=' * 80}\n")

        # Re-raise for application to handle
        raise


# Create global settings instance
settings = get_settings()