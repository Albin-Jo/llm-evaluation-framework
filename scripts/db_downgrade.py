# File: scripts/db_downgrade.py
# !/usr/bin/env python
import subprocess
import os
import argparse


def downgrade_database(revision):
    """Downgrade the database to the specified revision."""
    # Use production settings but could be configured per environment
    os.environ["APP_ENV"] = os.getenv("APP_ENV", "production")

    try:
        subprocess.run(["alembic", "downgrade", revision], check=True)
        print(f"Database downgraded successfully to {revision}")
    except subprocess.CalledProcessError as e:
        print(f"Error downgrading database: {e}")
        return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downgrade database to a revision")
    parser.add_argument(
        "revision",
        help="Revision to downgrade to (e.g., '-1' for previous, or specific revision ID)"
    )

    args = parser.parse_args()
    downgrade_database(args.revision)