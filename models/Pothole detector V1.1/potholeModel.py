from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

WEIGHTS_PATH = (
    BASE_DIR
    / "runs"
    / "detect"
    / "pothole_yolov8"
    / "weights"
    / "best.pt"
)

if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(
        f"Pothole model weights not found: {WEIGHTS_PATH}"
    )

model = YOLO(WEIGHTS_PATH)