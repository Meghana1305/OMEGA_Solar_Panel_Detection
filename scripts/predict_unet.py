
from pathlib import Path
import time

import cv2
import numpy as np
import psutil
import torch
import torch.nn as nn


# ---------------- CONFIGURATION ----------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

TEST_DIR = (
    PROJECT_DIR / "data" / "processed" / "PV03_split" / "test"
)

MODEL_PATH = (
    PROJECT_DIR / "outputs" / "models" / "unet_pv03_best.pth"
)

OUTPUT_DIR = PROJECT_DIR / "outputs" / "predictions"

MASK_DIR = OUTPUT_DIR / "masks"
OVERLAY_DIR = OUTPUT_DIR / "overlays"
COMPARE_DIR = OUTPUT_DIR / "comparisons"

for folder in [MASK_DIR, OVERLAY_DIR, COMPARE_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")
torch.set_num_threads(2)


# ---------------- U-NET MODEL ----------------

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, padding=1
        ),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, padding=1
        ),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.enc1 = double_conv(3, 8)
        self.enc2 = double_conv(8, 16)
        self.enc3 = double_conv(16, 32)
        self.enc4 = double_conv(32, 64)

        self.bottleneck = double_conv(64, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec4 = double_conv(128, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec3 = double_conv(64, 32)

        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec2 = double_conv(32, 16)

        self.up1 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.dec1 = double_conv(16, 8)

        self.output = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.output(d1)


# ---------------- IMAGE PREPROCESSING ----------------

def load_image(image_path, image_size):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(
            f"Could not read image: {image_path}"
        )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(
        image,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA
    )

    image_float = image.astype(np.float32) / 255.0

    tensor = torch.from_numpy(image_float)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    return image, tensor


def load_mask(mask_path, image_size):
    mask = cv2.imread(
        str(mask_path), cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        raise RuntimeError(
            f"Could not read mask: {mask_path}"
        )

    mask = cv2.resize(
        mask,
        (image_size, image_size),
        interpolation=cv2.INTER_NEAREST
    )

    return (mask > 0).astype(np.uint8)


# ---------------- MAIN EVALUATION ----------------

def main():
    print("Device:", DEVICE)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    image_dir = TEST_DIR / "images"
    mask_dir = TEST_DIR / "masks"

    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Test images or masks folder is missing: {TEST_DIR}"
        )

    # Load the saved model checkpoint.
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    image_size = checkpoint["image_size"]

    model = UNet().to(DEVICE)
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )
    model.eval()

    process = psutil.Process()

    print("Model loaded successfully.")
    print("Image size:", image_size)

    image_paths = sorted(
        path for path in image_dir.iterdir()
        if path.is_file()
        and not path.stem.endswith("_label")
    )

    dice_scores = []
    iou_scores = []
    inference_times = []

    processed = 0
    skipped = 0

    # Begin resource monitoring.
    process.cpu_percent(interval=None)
    cpu_samples = []

    # Warm up the model before timing.
    warmup_tensor = torch.zeros(
        (1, 3, image_size, image_size),
        dtype=torch.float32
    )

    with torch.no_grad():
        for _ in range(2):
            model(warmup_tensor)

    print("\nEvaluating test dataset...")

    evaluation_start = time.perf_counter()

    with torch.no_grad():
        for index, image_path in enumerate(
            image_paths, start=1
        ):
            mask_path = mask_dir / (
                image_path.stem
                + "_label"
                + image_path.suffix
            )

            if not mask_path.exists():
                skipped += 1
                continue

            original, image_tensor = load_image(
                image_path, image_size
            )

            ground_truth = load_mask(
                mask_path, image_size
            )

            # Measure model inference time only.
            start_time = time.perf_counter()

            logits = model(image_tensor)
            probabilities = torch.sigmoid(logits)

            prediction = (
                probabilities[0, 0].numpy() >= 0.5
            ).astype(np.uint8)

            inference_time = (
                time.perf_counter() - start_time
            )
            inference_times.append(inference_time)

            # Calculate Dice Score.
            intersection = np.logical_and(
                prediction, ground_truth
            ).sum()

            pred_sum = prediction.sum()
            gt_sum = ground_truth.sum()

            dice = (
                2.0 * intersection / (pred_sum + gt_sum)
                if pred_sum + gt_sum > 0
                else 1.0
            )

            # Calculate Intersection over Union.
            union = np.logical_or(
                prediction, ground_truth
            ).sum()

            iou = (
                intersection / union
                if union > 0
                else 1.0
            )

            dice_scores.append(dice)
            iou_scores.append(iou)

            # Save predicted binary mask.
            predicted_mask = prediction * 255

            cv2.imwrite(
                str(
                    MASK_DIR
                    / f"{image_path.stem}_pred.png"
                ),
                predicted_mask
            )

            # Convert original image to BGR for saving.
            original_bgr = cv2.cvtColor(
                original, cv2.COLOR_RGB2BGR
            )

            # Save green prediction overlay.
            overlay = original_bgr.copy()

            overlay[prediction == 1] = (
                0.5 * overlay[prediction == 1]
                + 0.5 * np.array([0, 255, 0])
            ).astype(np.uint8)

            cv2.imwrite(
                str(
                    OVERLAY_DIR
                    / f"{image_path.stem}_overlay.png"
                ),
                overlay
            )

            # Save original | ground truth | prediction.
            gt_panel = original_bgr.copy()

            gt_panel[ground_truth == 1] = (
                0.5 * gt_panel[ground_truth == 1]
                + 0.5 * np.array([255, 0, 0])
            ).astype(np.uint8)

            comparison = np.hstack([
                original_bgr,
                gt_panel,
                overlay
            ])

            cv2.imwrite(
                str(
                    COMPARE_DIR
                    / f"{image_path.stem}_compare.png"
                ),
                comparison
            )

            processed += 1

            # Collect CPU usage samples periodically.
            if processed % 10 == 0:
                cpu_samples.append(
                    process.cpu_percent(interval=None)
                )

            if processed % 25 == 0:
                print(
                    f"Processed {processed} test images"
                )

    total_evaluation_time = (
        time.perf_counter() - evaluation_start
    )

    if processed == 0:
        raise RuntimeError(
            "No test image-mask pairs found. "
            "Check image and mask filenames."
        )

    # ---------------- FINAL RESULTS ----------------

    mean_dice = float(np.mean(dice_scores))
    mean_iou = float(np.mean(iou_scores))
    mean_inference_time = float(
        np.mean(inference_times)
    )

    inference_fps = (
        1.0 / mean_inference_time
        if mean_inference_time > 0
        else 0.0
    )

    total_ram_mb = (
        process.memory_info().rss / (1024 ** 2)
    )

    average_cpu = (
        float(np.mean(cpu_samples))
        if cpu_samples
        else None
    )

    peak_cpu = (
        float(np.max(cpu_samples))
        if cpu_samples
        else None
    )

    print("\n========== TEST RESULTS ==========")
    print(f"Test image-mask pairs: {processed}")
    print(f"Skipped images: {skipped}")
    print(f"Test Dice Score: {mean_dice:.4f}")
    print(f"Test IoU: {mean_iou:.4f}")

    print("\n========== INFERENCE PERFORMANCE ==========")
    print(
        f"Average model inference time: "
        f"{mean_inference_time:.4f} seconds/image"
    )
    print(f"Approximate model inference FPS: {inference_fps:.2f}")
    print(
        f"Total evaluation time: "
        f"{total_evaluation_time:.2f} seconds"
    )

    print("\n========== RESOURCE USAGE ==========")
    print(f"Process RAM at end: {total_ram_mb:.2f} MB")

    if average_cpu is not None:
        print(f"Average sampled process CPU: {average_cpu:.2f}%")
        print(f"Peak sampled process CPU: {peak_cpu:.2f}%")
    else:
        print(
            "CPU samples unavailable for this run."
        )

    print("\n========== SAVED RESULTS ==========")
    print("Predicted masks:", MASK_DIR)
    print("Green overlays:", OVERLAY_DIR)
    print("Comparison images:", COMPARE_DIR)

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
