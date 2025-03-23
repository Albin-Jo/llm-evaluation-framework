# File: fix_storage_path.py
import os
from pathlib import Path

# Create the required directories
storage_path = Path("storage/datasets")
os.makedirs(storage_path, exist_ok=True)

print(f"Created directory: {storage_path.absolute()}")

# Create a test file to verify write permissions
test_file = storage_path / "test_file.txt"
with open(test_file, "w") as f:
    f.write("Test file to verify storage directory works")

print(f"Created test file: {test_file.absolute()}")
print("Storage directory is now ready for use")