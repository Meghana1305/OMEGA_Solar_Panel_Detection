
from pathlib import Path
import cv2
import numpy as np

project_dir = Path(__file__).resolve().parent.parent
mask_dir = project_dir / "data" / "processed" / "PV03_split" / "train" / "masks"

mask_files = sorted(mask_dir.glob("*"))

if not mask_files:
    raise RuntimeError(f"No masks found in {mask_dir}")

for mask_path in mask_files[:5]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

    if mask is None:
        print(f"Could not read: {mask_path.name}")
        continue

    values, counts = np.unique(mask, return_counts=True)

    print(f"\nMask: {mask_path.name}")
    print("Shape:", mask.shape)
    print("Data type:", mask.dtype)
    print("Unique pixel values:", values[:30])
    print("Pixel counts:", counts[:30])
