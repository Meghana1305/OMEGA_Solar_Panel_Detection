
from pathlib import Path
from collections import Counter

DATASET_DIR = Path("data/raw/PV03")

image_extensions = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
total_images = 0
total_masks = 0
paired = 0
unpaired = []
folders = Counter()

for folder in DATASET_DIR.iterdir():
    if not folder.is_dir():
        continue

    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    masks = {
        p.stem.removesuffix("_label").lower(): p
        for p in files
        if "_label" in p.stem.lower()
    }

    images = [
        p for p in files
        if "_label" not in p.stem.lower()
    ]

    total_images += len(images)
    total_masks += len(files) - len(images)

    folder_pairs = 0

    for image_path in images:
        if image_path.stem.lower() in masks:
            paired += 1
            folder_pairs += 1
        else:
            unpaired.append(str(image_path))

    folders[folder.name] = folder_pairs
    print(
        f"{folder.name}: "
        f"{len(images)} images, "
        f"{len(masks)} masks, "
        f"{folder_pairs} pairs"
    )

print("\n--- DATASET AUDIT ---")
print("Image files:", total_images)
print("Mask files:", total_masks)
print("Matched pairs:", paired)
print("Images without matching masks:", len(unpaired))

if unpaired:
    print("\nFirst 10 unmatched images:")
    for path in unpaired[:10]:
        print(path)
