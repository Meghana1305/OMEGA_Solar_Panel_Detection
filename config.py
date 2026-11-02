from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw"

DATASETS = {
    "PV01": DATA_ROOT / "PV01",
    "PV03": DATA_ROOT / "PV03",
    "PV08": DATA_ROOT / "PV08",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_HINTS = ("mask", "label", "annotation", "groundtruth", "gt")
