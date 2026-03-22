import os
import shutil

from config import (
    DISTANCE_THRESHOLD,
    MERGE_SUGGESTION_THRESHOLD,
    MODELS_FOLDER,
    OUTPUT_FOLDER,
)
from organizer import organize_photos

if __name__ == "__main__":
    # Remove existing output folder for fresh run
    if os.path.exists(OUTPUT_FOLDER):
        shutil.rmtree(OUTPUT_FOLDER)

    # Run the organization
    organize_photos(
        MODELS_FOLDER, OUTPUT_FOLDER, DISTANCE_THRESHOLD, MERGE_SUGGESTION_THRESHOLD
    )
