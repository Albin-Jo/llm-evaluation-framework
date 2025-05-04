"""
Script to create Alembic migrations for FastAPI project.
This script should be run from the project root directory.
"""
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path


def setup_logging():
    """Configure logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


def create_migration(message, environment="development", logger=None):
    """
    Create a new Alembic migration.

    Args:
        message (str): Migration message
        environment (str): Environment to use (development, testing, production)
        logger: Logger instance

    Returns:
        bool: True if migration was created successfully, False otherwise
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Check if alembic.ini exists in current directory
    if not Path("alembic.ini").exists():
        logger.error("alembic.ini not found. Please run from project root directory.")
        return False

    # Set environment
    os.environ["APP_ENV"] = environment
    logger.info(f"Creating migration in {environment} environment")

    # Run Alembic revision
    try:
        result = subprocess.run(
            ["alembic", "revision", "--autogenerate", "-m", message],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info(f"Successfully created migration with message: '{message}'")
        logger.debug(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error creating migration: {e}")
        if e.stdout:
            logger.error(f"Output: {e.stdout}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False


if __name__ == "__main__":
    logger = setup_logging()

    parser = argparse.ArgumentParser(description="Create a new database migration")
    parser.add_argument(
        "message",
        help="Migration message (e.g., 'add_user_table')"
    )
    parser.add_argument(
        "--env", "-e",
        default="development",
        choices=["development", "testing", "production"],
        help="Environment to use (default: development)"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    # Confirm if not using --yes flag
    if not args.yes and args.env == "production":
        confirm = input(f"Are you sure you want to create a migration in {args.env} environment? (y/N): ")
        if confirm.lower() != "y":
            logger.info("Operation canceled.")
            sys.exit(0)

    success = create_migration(args.message, args.env, logger)
    sys.exit(0 if success else 1)