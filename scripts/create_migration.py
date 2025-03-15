# File: scripts/create_migration.py
# !/usr/bin/env python
import argparse
import os
import subprocess


def create_migration(message):
    """Create a new Alembic migration."""
    # Make sure we're using the right environment settings
    os.environ["APP_ENV"] = "development"

    # Run Alembic revision
    try:
        subprocess.run(
            ["alembic", "revision", "--autogenerate", "-m", message],
            check=True
        )
        print(f"Successfully created migration with message: '{message}'")
    except subprocess.CalledProcessError as e:
        print(f"Error creating migration: {e}")
        return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a new database migration")
    parser.add_argument(
        "message",
        help="Migration message (e.g., 'add_user_table')"
    )

    args = parser.parse_args()
    create_migration(args.message)