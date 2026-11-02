
from pathlib import Path
import cv2
import numpy as np

# ---------------- CONFIGURATION ----------------
ROOT = Path(r"E:\dowloads\OMEGA_Solar_Panel_Project")
DATASET = ROOT / "data" / "processed" / "PV03_split"
OUTPUT_DIR = ROOT / "outputs" / "pair_comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MASK_KEYWORDS = ("mask", "label", "groundtruth", "ground_truth", "gt")

print("=" * 60)
print("OMEGA - PV03 IMAGE/MASK COMPARISON")
print("=" * 60)
print("Dataset:", DATASET)
print("Dataset exists:", DATASET.exists())
print("Output:", OUTPUT_DIR)

if not DATASET.exists():
    print("\nERROR: Dataset folder not found.")
    raise SystemExit(1)

# ---------------- FIND FILES ----------------
all_files = [
    p for p in DATASET.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
]

print("\nTotal image files:", len(all_files))

if not all_files:
    print("ERROR: No supported image files found.")
    raise SystemExit(1)

print("\nSample filenames:")
for path in all_files[:20]:
    print(" ", path.relative_to(DATASET))

# Identify mask files using filenames or parent folder names
mask_files = []

for path in all_files:
    name = path.stem.lower()
    parent_names = " ".join(part.lower() for part in path.parts)

    if any(keyword in name for keyword in MASK_KEYWORDS):
        mask_files.append(path)
    elif any(
        folder in parent_names.split()
        for folder in ["masks", "labels", "groundtruth", "ground_truth"]
    ):
        mask_files.append(path)

mask_files = sorted(set(mask_files))

print("\nPossible mask files:", len(mask_files))

# ---------------- MATCH IMAGE TO MASK ----------------
def possible_image_names(mask_path):
    stem = mask_path.stem

    suffixes = [
        "_label", "-label", " label",
        "_mask", "-mask", " mask",
        "_groundtruth", "_ground_truth", "_gt"
    ]

    candidates = {stem}

    for suffix in suffixes:
        if stem.lower().endswith(suffix):
            candidates.add(stem[:-len(suffix)])

    # Also try removing a trailing "label" without a separator
    if stem.lower().endswith("label"):
        candidates.add(stem[:-5].rstrip("_- "))

    return candidates


def find_matching_image(mask_path):
    candidates = possible_image_names(mask_path)

    # First try the same folder
    for candidate in candidates:
        for ext in IMAGE_EXTENSIONS:
            image_path = mask_path.with_name(candidate + ext)
            if image_path.exists() and image_path != mask_path:
                return image_path

    # Then search the dataset for the same base filename
    for image_path in all_files:
        if image_path == mask_path:
            continue

        if image_path.stem in candidates:
            # Avoid choosing another mask
            if not any(
                keyword in image_path.stem.lower()
                for keyword in MASK_KEYWORDS
            ):
                return image_path

    return None


# ---------------- CREATE COMPARISONS ----------------
created = 0
unmatched = 0
MAX_PAIRS = 5

for mask_path in mask_files:
    image_path = find_matching_image(mask_path)

    if image_path is None:
        unmatched += 1
        if unmatched <= 10:
            print("\nNo matching image for mask:", mask_path.name)
        continue

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print("\nCould not read pair:", image_path.name, mask_path.name)
        continue

    # Resize image and mask while preserving mask labels
    image = cv2.resize(image, (512, 512))
    mask = cv2.resize(
        mask, (512, 512), interpolation=cv2.INTER_NEAREST
    )

    binary_mask = (mask > 0).astype(np.uint8)

    # Green overlay marks foreground pixels
    overlay = image.copy()
    green = np.zeros_like(image)
    green[:, :] = (0, 255, 0)

    foreground = binary_mask == 1
    overlay[foreground] = (
        0.5 * image[foreground].astype(np.float32)
        + 0.5 * green[foreground].astype(np.float32)
    ).astype(np.uint8)

    mask_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Left: original | Middle: mask | Right: overlay
    comparison = np.hstack([image, mask_view, overlay])

    output_path = OUTPUT_DIR / f"comparison_{created + 1}.jpg"
    cv2.imwrite(str(output_path), comparison)

    print(f"\nPair {created + 1}")
    print("Image:", image_path.relative_to(DATASET))
    print("Mask :", mask_path.relative_to(DATASET))
    print("Saved:", output_path)

    created += 1

    if created >= MAX_PAIRS:
        break

print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
print("Comparisons created:", created)
print("Unmatched masks encountered:", unmatched)
print("Output folder:", OUTPUT_DIR)

if created == 0:
    print("\nNo pairs were created.")
    print("Use the sample filenames printed above to check")
    print("how your image and mask files are named and organized.")
else:
    print("\nOpen the comparison images in the output folder.")
