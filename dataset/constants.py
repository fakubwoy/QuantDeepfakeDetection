import os

DATA_FOLDER = os.path.join("..", "data")

# CORRECTED: Point to the top-level directories
ORIGINAL_VIDEOS_FOLDER = os.path.join(
    DATA_FOLDER, "original_sequences"
)
FAKE_VIDEOS_FOLDER = os.path.join(
    DATA_FOLDER, "manipulated_sequences"
)

# These paths are correct
FACES_FOLDER = os.path.join(DATA_FOLDER, "faces")
FACES_REAL = os.path.join(FACES_FOLDER, "real")
FACES_FAKE = os.path.join(FACES_FOLDER, "fake")
FACES_CSV = os.path.join(FACES_FOLDER, "faces.csv")