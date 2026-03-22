import os
from pathlib import Path

import cv2

from config import SUPPORTED_EXTENSIONS


def load_images(folder_path):
    """Load all images from the specified folder."""
    images = []
    image_files = []

    for file_name in sorted(os.listdir(folder_path)):
        ext = Path(file_name).suffix.lower()
        if ext in SUPPORTED_EXTENSIONS:
            file_path = os.path.join(folder_path, file_name)
            try:
                image = cv2.imread(file_path)
                if image is not None:
                    images.append(image)
                    image_files.append(file_name)
                else:
                    print(f"Warning: Could not read {file_name}")
            except Exception as e:
                print(f"Warning: Error loading {file_name}: {e}")

    return images, image_files
