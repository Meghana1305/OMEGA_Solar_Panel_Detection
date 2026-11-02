
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

UNET_PATH = BASE_DIR / "outputs" / "models" / "unet_pv03_best.pth"
YOLO_PATH = BASE_DIR / "outputs" / "models" / "yolo_solar_best.pt"

DEVICE = torch.device("cpu")

IMAGE_SIZE = 128

# Post-processing settings; these need validation on real images.
SEGMENTATION_THRESHOLD = 0.50
MIN_COMPONENT_AREA = 8

torch.set_num_threads(2)

st.set_page_config(
    page_title="OMEGA | Solar Detection & Segmentation",
    page_icon="☀️",
    layout="wide",
)

st.title("☀️ OMEGA")
st.subheader("Solar Panel Detection & Image Segmentation")
st.write(
    "Upload an aerial or satellite image to identify solar-panel regions "
    "and visualize segmentation predictions."
)


# =========================================================
# 2. U-NET MODEL
# Must match the architecture used for training.
# =========================================================

class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2)

        self.enc1 = self.block(3, 8)
        self.enc2 = self.block(8, 16)
        self.enc3 = self.block(16, 32)
        self.enc4 = self.block(32, 64)

        self.bottleneck = self.block(64, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec4 = self.block(128, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec3 = self.block(64, 32)

        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec2 = self.block(32, 16)

        self.up1 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.dec1 = self.block(16, 8)

        self.output = nn.Conv2d(8, 1, kernel_size=1)

    @staticmethod
    def block(in_channels, out_channels):
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
            nn.ReLU(inplace=True),
        )

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


# =========================================================
# 3. LOAD U-NET CHECKPOINT
# =========================================================

@st.cache_resource
def load_unet():

    if not UNET_PATH.exists():
        raise FileNotFoundError(
            f"U-Net model not found: {UNET_PATH}"
        )

    model = UNet()

    checkpoint = torch.load(
        UNET_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    # Support either a plain state dictionary or a checkpoint.
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


# =========================================================
# 4. LOAD YOLO MODEL, IF AVAILABLE
# =========================================================

@st.cache_resource
def load_yolo():

    if YOLO is None:
        return None, "Ultralytics is not installed."

    if not YOLO_PATH.exists():
        return None, "Trained YOLO checkpoint is missing."

    try:
        detector = YOLO(str(YOLO_PATH))
        return detector, "Trained YOLO checkpoint loaded."

    except Exception as exc:
        return None, f"YOLO model could not be loaded: {exc}"


# =========================================================
# 5. SEGMENTATION POST-PROCESSING
# =========================================================

def clean_mask(probability):

    # Threshold the predicted probability map.
    binary = (
        probability >= SEGMENTATION_THRESHOLD
    ).astype(np.uint8)

    # Remove small isolated components.
    # This can remove small genuine panels too.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    cleaned = np.zeros_like(binary)

    for i in range(1, count):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= MIN_COMPONENT_AREA:
            cleaned[labels == i] = 1

    return cleaned


def predict_segmentation(original, model):

    height, width = original.shape[:2]

    resized = cv2.resize(
        original,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    # Training used RGB pixels scaled to [0, 1].
    tensor = torch.from_numpy(
        resized.astype(np.float32) / 255.0
    ).permute(2, 0, 1).unsqueeze(0)

    tensor = tensor.to(DEVICE)

    with torch.inference_mode():
        logits = model(tensor)
        probability = torch.sigmoid(logits)[0, 0].cpu().numpy()

    small_mask = clean_mask(probability)

    # Resize the binary mask back to original image size.
    mask = cv2.resize(
        small_mask,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )

    return mask, probability


# =========================================================
# 6. CREATE SEGMENTATION OVERLAY
# =========================================================

def create_overlay(original, mask):

    overlay = original.copy()

    foreground = mask == 1

    green = np.array(
        [0, 255, 80],
        dtype=np.float32,
    )

    overlay[foreground] = (
        0.45 * original[foreground].astype(np.float32)
        + 0.55 * green
    ).clip(0, 255).astype(np.uint8)

    return overlay


# =========================================================
# 7. MASK-DERIVED CANDIDATE BOXES
# These are NOT YOLO detections.
# =========================================================

def boxes_from_mask(mask):

    binary = (mask > 0).astype(np.uint8)

    count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    boxes = []

    height, width = mask.shape

    for i in range(1, count):

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])

        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < 100:
            continue

        if w < 10 or h < 10:
            continue

        # Reject regions spanning nearly the entire image.
        if w > width * 0.95 and h > height * 0.95:
            continue

        boxes.append((x, y, x + w, y + h))

    return boxes


# =========================================================
# 8. DETECTION
# =========================================================

def make_detection_image(original, mask, detector):

    result = original.copy()

    if detector is not None:

        predictions = detector.predict(
            source=original,
            conf=0.25,
            imgsz=640,
            device="cpu",
            verbose=False,
        )

        pred = predictions[0]
        boxes = pred.boxes

        count = 0

        if boxes is not None:

            for box in boxes.xyxy.cpu().numpy().astype(int):

                x1, y1, x2, y2 = box

                x1 = max(0, x1)
                y1 = max(0, y1)

                x2 = min(original.shape[1] - 1, x2)
                y2 = min(original.shape[0] - 1, y2)

                cv2.rectangle(
                    result,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                count += 1

        return result, count, "YOLO model"

    # Fallback only: connected regions in the segmentation mask.
    candidate_boxes = boxes_from_mask(mask)

    for x1, y1, x2, y2 in candidate_boxes:

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            (255, 165, 0),
            2,
        )

    return (
        result,
        len(candidate_boxes),
        "Mask-derived candidate regions (not YOLO)",
    )


# =========================================================
# 9. IMAGE UPLOAD
# =========================================================

if not UNET_PATH.exists():

    st.error(
        f"U-Net checkpoint not found:\n{UNET_PATH}"
    )
    st.stop()

try:
    unet = load_unet()

except Exception as exc:

    st.error(f"Could not load U-Net checkpoint: {exc}")
    st.stop()

detector, yolo_status = load_yolo()

uploaded_file = st.file_uploader(
    "Upload an aerial or satellite image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
)


# =========================================================
# 10. RUN INFERENCE
# =========================================================

if uploaded_file is not None:

    try:

        pil_image = Image.open(uploaded_file).convert("RGB")
        original = np.array(pil_image)

        st.image(
            original,
            caption="Uploaded original image",
            width="stretch",
        )

        if st.button(
            "☀️ Detect and Segment",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Running segmentation and detection..."
            ):

                mask, probability = predict_segmentation(
                    original,
                    unet,
                )

                overlay = create_overlay(
                    original,
                    mask,
                )

                (
                    detection_image,
                    detection_count,
                    detection_mode,
                ) = make_detection_image(
                    original,
                    mask,
                    detector,
                )

                st.session_state["omega_results"] = {
                    "original": original,
                    "mask": mask * 255,
                    "overlay": overlay,
                    "detection": detection_image,
                    "count": detection_count,
                    "mode": detection_mode,
                    "yolo_status": yolo_status,
                }

    except Exception as exc:

        st.error(f"Could not process image: {exc}")


# =========================================================
# 11. DISPLAY RESULTS
# =========================================================

if "omega_results" in st.session_state:

    result = st.session_state["omega_results"]

    st.markdown("---")
    st.header("OMEGA Results")

    if detector is None:

        st.warning(
            "YOLO detection is unavailable. "
            + result["yolo_status"]
            + " The orange boxes, if shown, are candidate regions "
              "derived from the segmentation mask, not YOLO predictions."
        )

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            result["detection"],
            caption="Detection result",
            width="stretch",
        )

    with col2:

        st.image(
            result["overlay"],
            caption="U-Net segmentation overlay",
            width="stretch",
        )

    col3, col4 = st.columns(2)

    with col3:

        st.image(
            result["mask"],
            caption="Predicted segmentation mask",
            clamp=True,
            width="stretch",
        )

    with col4:

        st.image(
            result["original"],
            caption="Original image",
            width="stretch",
        )

    st.write(f"Detection mode: **{result['mode']}**")
    st.write(f"Detected/candidate regions: **{result['count']}**")

    # Segmentation foreground percentage.
    mask_array = result["mask"]

    foreground_percentage = (
        np.count_nonzero(mask_array)
        / mask_array.size
    ) * 100

    st.write(
        f"Predicted foreground coverage: "
        f"**{foreground_percentage:.2f}%**"
    )

    st.caption(
        "Foreground coverage is the percentage of pixels predicted "
        "as foreground. It is not a model accuracy score."
    )

    # =====================================================
    # 12. DOWNLOAD RESULTS
    # =====================================================

    st.markdown("---")
    st.subheader("Download Results")

    def png_bytes(array):

        buffer = BytesIO()

        Image.fromarray(array).save(
            buffer,
            format="PNG",
        )

        return buffer.getvalue()

    d1, d2, d3 = st.columns(3)

    with d1:

        st.download_button(
            "Download mask",
            data=png_bytes(result["mask"]),
            file_name="omega_mask.png",
            mime="image/png",
            use_container_width=True,
        )

    with d2:

        st.download_button(
            "Download overlay",
            data=png_bytes(result["overlay"]),
            file_name="omega_overlay.png",
            mime="image/png",
            use_container_width=True,
        )

    with d3:

        st.download_button(
            "Download detection",
            data=png_bytes(result["detection"]),
            file_name="omega_detection.png",
            mime="image/png",
            use_container_width=True,
        )

st.markdown("---")
st.caption(
    "OMEGA | Solar Panel Detection & Image Segmentation System"
)
