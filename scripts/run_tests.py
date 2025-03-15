# File: scripts/run_tests.py
# !/usr/bin/env python
import subprocess
import sys
import os


def run_tests_with_coverage():
    """Run tests with coverage report."""
    # Set environment variable for testing
    os.environ["APP_ENV"] = "testing"

    # Command to run pytest with coverage
    cmd = [
        "python", "-m", "pytest",
        "--cov=app",
        "--cov-report=term",
        "--cov-report=html:coverage_html",
        "-v"
    ]

    # Add arguments passed to this script
    cmd.extend(sys.argv[1:])

    # Run the command
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    run_tests_with_coverage()