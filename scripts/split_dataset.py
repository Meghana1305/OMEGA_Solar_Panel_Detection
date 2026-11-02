
from pathlib import Path
import random
import shutil

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_DIR / "data" / "raw" / "PV03"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed" / "PV03_split"

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
SEED = 42

random.seed(SEED)

image_extensions = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
pairs = []

# Find images and their matching masks
for folder in SOURCE_DIR.iterdir():
    if not folder.is_dir():
        continue

    for image_path in folder.iterdir():
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in image_extensions:
            continue
        if image_path.stem.endswith("_label"):
            continue

        mask_path = image_path.with_name(
            f"{image_path.stem}_label{image_path.suffix}"
        )

        if mask_path.exists():
            pairs.append((image_path, mask_path))

if not pairs:
    raise RuntimeError("No matching image-mask pairs found.")

random.shuffle(pairs)

total = len(pairs)
train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

splits = {
    "train": pairs[:train_end],
    "val": pairs[train_end:val_end],
    "test": pairs[val_end:],
}

# Stop if the output already exists to avoid overwriting data
if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"{OUTPUT_DIR} already exists. Rename or remove it only "
        "after checking its contents."
    )

try:
    for split_name, split_pairs in splits.items():
        image_dir = OUTPUT_DIR / split_name / "images"
        mask_dir = OUTPUT_DIR / split_name / "masks"

        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        for image_path, mask_path in split_pairs:
            shutil.copy2(image_path, image_dir / image_path.name)
            shutil.copy2(mask_path, mask_dir / mask_path.name)

        print(f"{split_name}: {len(split_pairs)} image-mask pairs")

    print(f"\nTotal pairs: {total}")
    print(f"Output folder: {OUTPUT_DIR}")

except Exception:
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    raise
