import os
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
MODELS_FOLDER = os.path.join(SCRIPT_DIR, "Models")
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "Organized_People")

# Algorithm parameters
# Stricter distance threshold = lower = more clusters (fewer false merges)
DISTANCE_THRESHOLD = 0.7

# Merge suggestion threshold (cosine similarity)
# Clusters with similarity above this will be suggested for manual review
MERGE_SUGGESTION_THRESHOLD = 0.65

# Face detection settings
DETECTION_SIZE = (800, 800)

# Clustering method: 'average', 'complete', 'single', 'ward'
METHOD = "average"
# larger = better quality but slower
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# Output file names
MERGE_SUGGESTIONS_FILE = "merge_suggestions.json"
ANALYSIS_FILE = "analysis.json"
