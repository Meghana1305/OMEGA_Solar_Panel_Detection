import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import DATASETS
from src.dataset_utils import find_pairs

parser = argparse.ArgumentParser(description="Preview likely image-mask pairs.")
parser.add_argument("--dataset", choices=DATASETS.keys(), default="PV01")
parser.add_argument("--max-pairs", type=int, default=6)
args = parser.parse_args()

folder = DATASETS[args.dataset]
files, pairs = find_pairs(folder)
if not folder.exists():
    raise SystemExit(f"Dataset folder not found: {folder}")
if not pairs:
    print("No likely image-mask pairs detected automatically.")
    print("The dataset naming/layout may differ. Original files were not changed.")
    print(f"Supported image files found: {len(files)}")
    raise SystemExit(0)

for index, (image_path, mask_path) in enumerate(pairs[:args.max_pairs], start=1):
    try:
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(image)
        axes[0].set_title(f"Image: {image_path.name}")
        axes[1].imshow(np.array(mask), cmap="gray")
        axes[1].set_title(f"Possible mask: {mask_path.name}")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        print(f"{index}. IMAGE: {image_path}\n   MASK: {mask_path}")
        plt.show()
    except Exception as exc:
        print(f"Could not preview {image_path.name}: {exc}")
