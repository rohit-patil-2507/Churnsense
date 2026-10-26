import os
from pathlib import Path
import logging

# Configure logging format and level
logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")

# List of files and folders to be created directly in the root directory
list_of_files = [
    "app.py",
    "data/raw/.gitkeep",
    "notebooks/01_Eda.ipynb",
    "notebooks/02_model_building.ipynb",
    "models/best_model.pkl",
    "utils/__init__.py",
    "utils/data_utils.py",
    "utils/model_utils.py",
    ".gitignore",
    "requirements.txt",
    "README.md",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # Create directory if it does not exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")

    # Create empty file if it doesn't exist or is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass  # Create empty file
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")

if __name__ == "__main__":
    print("Files and directory structure created successfully in the root directory!")