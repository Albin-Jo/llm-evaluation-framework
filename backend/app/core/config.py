# File: app/core/config.py
import os
import logging
from typing import List, Optional, Union, Dict, Any

from pydantic import model_validator, Field, SecretStr, field_validator, AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings for the LLM Evaluation Framework."""
    # App settings
    APP_NAME: str = "LLM Evaluation Framework"
    APP_DESCRIPTION: str = "A framework for evaluating domain-specific AI agents"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = "development"  # Default to development for easier local setup
    APP_DEBUG: bool = True  # Default to True for development
    APP_SECRET_KEY: SecretStr = Field(default=SecretStr("dev_secret_key_change_me"))
    APP_BASE_URL: str = "http://localhost:8000"
    API_V1_STR: str = "/api/v1"

    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]

    # Database settings
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: SecretStr = Field(default=SecretStr("postgres"))
    POSTGRES_DB: str = "llm_evaluation"
    DATABASE_URI: Optional[str] = None  # Will be computed if not provided

    # Storage settings
    STORAGE_TYPE: str = "local"
    STORAGE_PATH: str = "storage"
    MAX_UPLOAD_SIZE_MB: int = 50

    # Azure OpenAI settings
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_KEY: Optional[SecretStr] = None
    AZURE_OPENAI_VERSION: str = "2023-05-15"
    AZURE_OPENAI_DEPLOYMENT: Optional[str] = None

    # Agent API settings
    AGENT_API_KEY: Optional[SecretStr] = None
    AGENT_REQUEST_TIMEOUT: float = 30.0

    # Evaluation settings
    MAX_CONCURRENT_EVALUATIONS: int = 5
    EVALUATION_TIMEOUT_SECONDS: int = 300  # 5 minutes

    # Celery settings (for background tasks)
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # Auth settings (OIDC)
    OIDC_DISCOVERY_URL: str = "https://example.auth0.com/.well-known/openid-configuration"
    OIDC_CLIENT_ID: str = "your_client_id"
    OIDC_CLIENT_SECRET: SecretStr = Field(default=SecretStr("your_client_secret"))
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # Redis settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: str = "6379"

    # Logging settings
    LOG_LEVEL: str = "INFO"

    # Model configuration
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

            if self.POSTGRES_PASSWORD.get_secret_value() == "postgres":
                raise ValueError("Default database password cannot be used in production")

            if self.OIDC_CLIENT_SECRET.get_secret_value() == "your_client_secret":
                raise ValueError("Default OIDC client secret cannot be used in production")

        return self

    @model_validator(mode='after')
    def ensure_database_uri(self) -> 'Settings':
        """Ensure DATABASE_URI is always set with a valid value."""
        # If DATABASE_URI is explicitly set, use it
        if self.DATABASE_URI:
            return self

        # For testing, use SQLite
        if self.APP_ENV == "testing":
            self.DATABASE_URI = "sqlite+aiosqlite:///:memory:"
        else:
            # Extract password value - need to handle it as SecretStr
            db_password = self.POSTGRES_PASSWORD.get_secret_value()

            # Build PostgreSQL connection string
            self.DATABASE_URI = (
                f"postgresql+asyncpg://{self.POSTGRES_USER}:{db_password}"
                f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
            )

        # Log the DATABASE_URI for debugging (excluding password)
        masked_uri = self.get_masked_database_uri()
        logger = logging.getLogger("settings")
        logger.debug(f"Using database URI: {masked_uri}")

        return self

    def get_masked_database_uri(self) -> str:
        """Return DATABASE_URI with password masked for safe logging."""
        if not self.DATABASE_URI:
            return "None"

        # Simple masking by replacing password portion
        parts = self.DATABASE_URI.split('@')
        if len(parts) > 1:
            auth_parts = parts[0].split(':')
            if len(auth_parts) > 2:
                masked = f"{auth_parts[0]}:****@{parts[1]}"
                return masked

        # If parsing fails, return a fully masked version
        return "****MASKED-CONNECTION-STRING****"

    @property
    def SQLALCHEMY_DATABASE_URI(self) -> str:
        """Backward compatibility property for existing code."""
        return self.DATABASE_URI

    def __str__(self) -> str:
        """Create a readable string representation without sensitive data."""
        sensitive_fields = [
            "POSTGRES_PASSWORD", "APP_SECRET_KEY", "AZURE_OPENAI_KEY",
            "OIDC_CLIENT_SECRET", "AGENT_API_KEY"
        ]

        output = ["Settings:"]
        for key, value in self.__dict__.items():
            if key == "DATABASE_URI":
                output.append(f"  {key}: {self.get_masked_database_uri()}")
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