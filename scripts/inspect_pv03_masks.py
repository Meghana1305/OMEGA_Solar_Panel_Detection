
from pathlib import Path
import cv2
import numpy as np

PROJECT_DIR = Path(r"E:\dowloads\OMEGA_Solar_Panel_Project")
DATASET_DIR = PROJECT_DIR / "data" / "processed" / "PV03_split"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "mask_inspection"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Dataset exists:", DATASET_DIR.exists())
print("Dataset path:", DATASET_DIR)

if not DATASET_DIR.exists():
    print("\nDataset folder not found.")
    print("Folders available inside data/processed:")

    parent = PROJECT_DIR / "data" / "processed"
    if parent.exists():
        for item in parent.iterdir():
            print(item)
    else:
        print("data/processed does not exist.")

else:
    print("\nDataset folders:")
    for item in DATASET_DIR.iterdir():
        print(item.name, " | Folder:", item.is_dir())

    mask_files = [
        p for p in DATASET_DIR.rglob("*")
        if p.is_file()
        and "mask" in str(p.parent).lower()
        and p.suffix.lower() in [".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"]
    ]

    print("\nMask files found:", len(mask_files))

    for i, mask_path in enumerate(mask_files[:5], start=1):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print("Could not read:", mask_path)
            continue

        binary = (mask > 0).astype(np.uint8)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        areas = stats[1:, cv2.CC_STAT_AREA]
        foreground = int(binary.sum())

        print(f"\nMask {i}: {mask_path.name}")
        print("Shape:", mask.shape)
        print("Pixel values:", np.unique(mask)[:20])
        print("Foreground pixels:", foreground)
        print("Connected regions:", count - 1)
        print("Largest region areas:", sorted(areas, reverse=True)[:10])

        preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        preview_path = OUTPUT_DIR / f"mask_preview_{i}.png"
        cv2.imwrite(str(preview_path), preview)

    print("\nPreview images saved to:", OUTPUT_DIR)
