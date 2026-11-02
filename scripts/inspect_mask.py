
from pathlib import Path
import cv2
import numpy as np

ROOT = Path("data/raw/PV03/PV03_Rooftop")
OUT = Path("outputs/mask_inspection")
OUT.mkdir(parents=True, exist_ok=True)

images = sorted(
    p for p in ROOT.glob("*.bmp")
    if "_label" not in p.stem.lower()
)

if not images:
    raise FileNotFoundError(f"No images found in {ROOT}")

image_path = images[0]
mask_path = image_path.with_name(
    image_path.stem + "_label.bmp"
)

if not mask_path.exists():
    raise FileNotFoundError(f"Matching mask not found: {mask_path}")

image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

if image is None or mask is None:
    raise ValueError("Could not read image or mask")

print("Image:", image_path.name)
print("Image shape:", image.shape)
print("Mask:", mask_path.name)
print("Mask shape:", mask.shape)
print("Mask data type:", mask.dtype)
print("Mask unique values:", np.unique(mask)[:30])

values, counts = np.unique(mask, return_counts=True)
print("Pixel counts:", dict(zip(values.tolist(), counts.tolist())))

# Save the original image
cv2.imwrite(str(OUT / "original.png"), image)

# Save the original mask without changing its values
if mask.ndim == 2:
    mask_preview = cv2.normalize(
        mask, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
else:
    mask_preview = mask

cv2.imwrite(str(OUT / "mask_preview.png"), mask_preview)

# Create a binary preview for visual inspection only
if mask.ndim == 2:
    binary = (mask > 0).astype(np.uint8) * 255
    cv2.imwrite(str(OUT / "binary_mask.png"), binary)

    overlay = image.copy()
    overlay[binary > 0] = (0, 255, 0)

    overlay = cv2.addWeighted(image, 0.65, overlay, 0.35, 0)
    cv2.imwrite(str(OUT / "overlay.png"), overlay)

print("\nSaved inspection files to:", OUT.resolve())
