
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(r"E:\dowloads\OMEGA_Solar_Panel_Project")
DATASET = ROOT / "data" / "processed" / "PV03_split"
OUTPUT = ROOT / "outputs" / "yolo_box_audit"

OUTPUT.mkdir(parents=True, exist_ok=True)

SPLIT = "train"
MIN_AREA = 100
MAX_SAMPLES = 8

image_dir = DATASET / SPLIT / "images"
mask_dir = DATASET / SPLIT / "masks"

mask_files = sorted(mask_dir.glob("*_label.bmp"))

print("Split:", SPLIT)
print("Images folder exists:", image_dir.exists())
print("Masks found:", len(mask_files))

if not image_dir.exists() or not mask_files:
    raise SystemExit("Could not find the expected image/mask folders.")

created = 0

for mask_path in mask_files:
    image_path = image_dir / mask_path.name.replace("_label.bmp", ".bmp")

    if not image_path.exists():
        continue

    image = cv2.imread(str(image_path))
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        continue

    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    boxes = []

    for i in range(1, count):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < MIN_AREA:
            continue

        boxes.append((x, y, w, h, area))

    preview = cv2.resize(image, (512, 512))
    scale_x = 512 / image.shape[1]
    scale_y = 512 / image.shape[0]

    for x, y, w, h, area in boxes:
        x1 = round(x * scale_x)
        y1 = round(y * scale_y)
        x2 = round((x + w) * scale_x)
        y2 = round((y + h) * scale_y)

        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 0, 255), 2)

    output_path = OUTPUT / f"{image_path.stem}_boxes.jpg"
    cv2.imwrite(str(output_path), preview)

    print(f"\n{image_path.name}: {len(boxes)} candidate boxes")
    for box in boxes[:10]:
        print("  x, y, width, height, area:", box)

    created += 1

    if created >= MAX_SAMPLES:
        break

print(f"\nCreated {created} box previews.")
print("Output folder:", OUTPUT)
print("These are candidate boxes, not verified YOLO labels.")
