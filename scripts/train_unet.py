
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------- CONFIGURATION ----------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed" / "PV03_split"
MODEL_DIR = PROJECT_DIR / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 128
BATCH_SIZE = 2
EPOCHS = 10
LEARNING_RATE = 0.001
PROGRESS_EVERY = 25

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(2)

DEVICE = torch.device("cpu")


# ---------------- DATASET ----------------

class SolarDataset(Dataset):
    def __init__(self, split):
        self.image_dir = DATA_DIR / split / "images"
        self.mask_dir = DATA_DIR / split / "masks"

        if not self.image_dir.exists() or not self.mask_dir.exists():
            raise FileNotFoundError(f"Missing image or mask folder: {split}")

        self.pairs = []

        for image_path in sorted(self.image_dir.iterdir()):
            if not image_path.is_file():
                continue
            if image_path.stem.endswith("_label"):
                continue

            mask_path = self.mask_dir / (
                image_path.stem + "_label" + image_path.suffix
            )

            if mask_path.exists():
                self.pairs.append((image_path, mask_path))

        if not self.pairs:
            raise RuntimeError(f"No image-mask pairs found for {split}")

        print(f"{split}: {len(self.pairs)} image-mask pairs", flush=True)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.pairs[index]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            raise RuntimeError(f"Could not read pair: {image_path.name}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        mask = cv2.resize(
            mask,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST
        )

        image = image.astype(np.float32) / 255.0

        # Binary foreground: all nonzero pixels become 1.
        mask = (mask > 0).astype(np.float32)

        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


# ---------------- U-NET MODEL ----------------

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
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


# ---------------- LOSS AND METRICS ----------------

def dice_loss(logits, targets):
    probabilities = torch.sigmoid(logits)
    dims = (1, 2, 3)

    intersection = (probabilities * targets).sum(dims)
    denominator = probabilities.sum(dims) + targets.sum(dims)

    dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
    return 1 - dice.mean()


def calculate_metrics(logits, targets):
    predictions = torch.sigmoid(logits) >= 0.5
    targets = targets >= 0.5

    dims = (1, 2, 3)

    intersection = (predictions & targets).sum(dims).float()
    pred_sum = predictions.sum(dims).float()
    target_sum = targets.sum(dims).float()
    union = (predictions | targets).sum(dims).float()

    dice = (2 * intersection + 1e-6) / (
        pred_sum + target_sum + 1e-6
    )
    iou = (intersection + 1e-6) / (union + 1e-6)

    return dice.mean().item(), iou.mean().item()


# ---------------- TRAINING ----------------

def main():
    print("Device:", DEVICE, flush=True)

    train_data = SolarDataset("train")
    val_data = SolarDataset("val")

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = UNet().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    bce_loss = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    model_path = MODEL_DIR / "unet_pv03_best.pth"

    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        train_loss_sum = 0.0

        total_batches = len(train_loader)

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS} started "
            f"({total_batches} training batches)",
            flush=True
        )

        for batch_index, (images, masks) in enumerate(train_loader):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)
            loss = bce_loss(logits, masks) + dice_loss(logits, masks)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()

            if (
                (batch_index + 1) % PROGRESS_EVERY == 0
                or batch_index + 1 == total_batches
            ):
                print(
                    f"  Batch {batch_index + 1}/{total_batches} | "
                    f"Loss: {loss.item():.4f}",
                    flush=True
                )

        model.eval()
        val_loss_sum = 0.0
        dice_sum = 0.0
        iou_sum = 0.0
        batch_count = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                logits = model(images)
                loss = bce_loss(logits, masks) + dice_loss(logits, masks)

                dice, iou = calculate_metrics(logits, masks)

                val_loss_sum += loss.item()
                dice_sum += dice
                iou_sum += iou
                batch_count += 1

        train_loss = train_loss_sum / total_batches
        val_loss = val_loss_sum / batch_count
        val_dice = dice_sum / batch_count
        val_iou = iou_sum / batch_count

        print(
            f"Epoch {epoch + 1}/{EPOCHS} complete | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Dice: {val_dice:.4f} | "
            f"Val IoU: {val_iou:.4f} | "
            f"Time: {time.time() - start_time:.1f}s",
            flush=True
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": IMAGE_SIZE,
                    "input_channels": 3,
                    "output_channels": 1,
                    "epoch": epoch + 1,
                    "val_loss": val_loss
                },
                model_path
            )

            print(f"  Best model saved: {model_path}", flush=True)

    print("\nTraining finished.", flush=True)
    print("Best model:", model_path, flush=True)


if __name__ == "__main__":
    main()
