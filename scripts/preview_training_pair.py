
from pathlib import Path
import cv2
import numpy as np

project_dir = Path(__file__).resolve().parent.parent
split_dir = project_dir / "data" / "processed" / "PV03_split" / "train"

mask_path = split_dir / "masks" / "PV03_314658_1199334_label.bmp"
image_path = split_dir / "images" / "PV03_314658_1199334.bmp"

image = cv2.imread(str(image_path))
mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

if image is None or mask is None:
    raise FileNotFoundError("Could not find the matching image and mask.")

# Convert all nonzero mask pixels to the foreground candidate.
binary_mask = (mask > 0).astype(np.uint8) * 255

# Create a colored overlay for visual inspection.
overlay = image.copy()
overlay[binary_mask > 0] = (0, 255, 0)
overlay = cv2.addWeighted(image, 0.65, overlay, 0.35, 0)

output_dir = project_dir / "outputs" / "mask_preview"
output_dir.mkdir(parents=True, exist_ok=True)

cv2.imwrite(str(output_dir / "original.png"), image)
cv2.imwrite(str(output_dir / "mask.png"), binary_mask)
cv2.imwrite(str(output_dir / "overlay.png"), overlay)

print("Preview images saved to:", output_dir)
