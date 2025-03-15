# File: scripts/db_upgrade.py
# !/usr/bin/env python
import subprocess
import os


def upgrade_database():
    """Upgrade the database to the latest version."""
    # Use production settings but could be configured per environment
    os.environ["APP_ENV"] = os.getenv("APP_ENV", "production")

    try:
        subprocess.run(["alembic", "upgrade", "head"], check=True)
        print("Database upgraded successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error upgrading database: {e}")
        return False

    return True


if __name__ == "__main__":
    upgrade_database()