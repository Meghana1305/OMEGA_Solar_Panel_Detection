
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
split_dir = project_dir / "data" / "processed" / "PV03_split"

for split in ["train", "val", "test"]:
    image_dir = split_dir / split / "images"
    mask_dir = split_dir / split / "masks"

    images = [
        p for p in image_dir.glob("*")
        if p.is_file() and not p.stem.endswith("_label")
    ]

    masks = [
        p for p in mask_dir.glob("*")
        if p.is_file()
    ]

    missing_masks = [
        image.name for image in images
        if not (mask_dir / f"{image.stem}_label{image.suffix}").exists()
    ]

    print(
        f"{split}: images={len(images)}, "
        f"masks={len(masks)}, "
        f"missing_masks={len(missing_masks)}"
    )

    if missing_masks:
        print("Examples missing masks:", missing_masks[:5])
