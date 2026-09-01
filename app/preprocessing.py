"""
preprocessing.py — OpenCV image preprocessing for lightweight PaddleOCR pipeline
==================================================================================

Provides a clean, testable preprocessing pipeline optimised for OCR on college
ID cards.  Separated from scanner logic so it can be imported and tested alone.

Functions
---------
validate_and_decode(image_bytes)
    Decode raw bytes → PIL Image.  Raises ValueError on bad input.

to_bgr(pil_img)
    Convert PIL RGB image → OpenCV BGR numpy array.

resize_for_ocr(bgr, max_side=1280)
    Resize image so its longest side ≤ max_side while preserving aspect ratio.

enhance_for_ocr(bgr)
    CLAHE contrast + mild unsharp-mask sharpening.
    Returns enhanced BGR array.

preprocess_pipeline(image_bytes, max_side=1280, save_debug=False, debug_tag="")
    Full pipeline: decode → resize → enhance.
    Returns dict with 'original', 'processed', and optionally 'debug_paths'.

rotate_image(bgr, angle)
    Rotate BGR image by exactly 0, 90, 180, or 270 degrees (lossless).

Debug image saving
------------------
Set env var  SAVE_DEBUG_IMAGES=1  to enable saving of intermediate images
to the  debug/  folder in the project root.
"""

from __future__ import annotations

import os
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

_DEBUG_ENABLED = os.environ.get("SAVE_DEBUG_IMAGES", "0").strip() == "1"
_DEBUG_DIR = Path(__file__).parent.parent / "debug"


def _save_debug(img: np.ndarray, label: str, tag: str = "") -> Optional[str]:
    """Save a BGR image to the debug folder.  Returns saved path or None."""
    if not _DEBUG_ENABLED:
        return None
    _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    ts = int(time.time() * 1000)
    suffix = f"_{tag}" if tag else ""
    path = _DEBUG_DIR / f"{label}{suffix}_{ts}.jpg"
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return str(path)


# ---------------------------------------------------------------------------
# Core preprocessing steps
# ---------------------------------------------------------------------------


def validate_and_decode(image_bytes: bytes) -> Image.Image:
    """
    Decode raw image bytes into a PIL Image (RGB).

    Raises
    ------
    ValueError
        If bytes are empty, not a valid image, or the format is unsupported.
    """
    if not image_bytes:
        raise ValueError("Image bytes are empty.")

    try:
        pil_img = Image.open(BytesIO(image_bytes))
        pil_img.verify()  # structural check — detects corrupt files
    except Exception as exc:
        raise ValueError(f"Invalid or corrupt image: {exc}") from exc

    # Re-open after verify() (verify() leaves the file in an unusable state)
    pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    return pil_img


def to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL RGB image to an OpenCV BGR numpy array."""
    return cv2.cvtColor(np.array(pil_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)


def resize_for_ocr(bgr: np.ndarray, max_side: int = 1280) -> np.ndarray:
    """
    Resize image so its longest side ≤ max_side, preserving aspect ratio.

    Does nothing if the image is already small enough.  Uses LANCZOS-equivalent
    (INTER_AREA for downscale, INTER_LINEAR for upscale — ID cards are rarely
    upscaled but we handle it gracefully).
    """
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return bgr  # no resize needed

    scale = max_side / longest
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(bgr, (new_w, new_h), interpolation=interp)


def enhance_for_ocr(bgr: np.ndarray) -> np.ndarray:
    """
    Contrast enhancement and mild sharpening for better OCR readability.

    Steps
    -----
    1. CLAHE on L-channel (LAB colour space) — improves contrast locally
       without blowing out already-bright areas.
    2. Mild unsharp mask on grayscale — sharpens text edges without creating
       ringing artefacts on very clean images.

    The CLAHE parameters (clipLimit=2.0, tileGridSize=8x8) are deliberately
    gentle: aggressive CLAHE can create false dark/light banding that confuses
    the text detector on short-text ID cards.
    """
    # --- CLAHE on L channel -----------------------------------------------
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_ch)
    lab_enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # --- Mild unsharp mask (sharpening) -----------------------------------
    # kernel_size=0 → auto-calculate from sigma; sigma=1.0 is mild
    blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=1.0)
    # amount=0.5: halfway between original and high-frequency boost
    result = cv2.addWeighted(result, 1.5, blurred, -0.5, 0)

    return result


def rotate_image(bgr: np.ndarray, angle: int) -> np.ndarray:
    """
    Rotate BGR image by exactly 0, 90, 180, or 270 degrees (lossless).

    Uses numpy rot90 (no interpolation loss, exact pixel rotation).
    Any angle not in {0, 90, 180, 270} is treated as 0 (no rotation).
    """
    if angle == 90:
        return np.rot90(bgr, k=3)   # counter-clockwise 270° == clockwise 90°
    elif angle == 180:
        return np.rot90(bgr, k=2)
    elif angle == 270:
        return np.rot90(bgr, k=1)   # counter-clockwise 90° == clockwise 270°
    return bgr  # 0°


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------


def preprocess_pipeline(
    image_bytes: bytes,
    max_side: int = 1280,
    save_debug: bool = False,
    debug_tag: str = "",
) -> dict:
    """
    Complete preprocessing pipeline for an ID card image.

    Parameters
    ----------
    image_bytes : bytes
        Raw image file bytes (JPEG / PNG / WEBP).
    max_side : int
        Resize limit — longest side in pixels.
    save_debug : bool
        If True (and SAVE_DEBUG_IMAGES=1), save intermediate images.
    debug_tag : str
        Short tag appended to debug image filenames.

    Returns
    -------
    dict with keys:
        'original_wh'   : (width, height) of the decoded image before resize
        'processed_wh'  : (width, height) after resize
        'processed'     : BGR numpy array — ready for OCR
        'debug_paths'   : list[str] of saved debug image paths (may be empty)

    Raises
    ------
    ValueError
        Propagated from validate_and_decode() if the image is invalid.
    """
    t0 = time.perf_counter()
    debug_paths: list[str] = []

    # Step 1: Decode
    pil_img = validate_and_decode(image_bytes)
    bgr_original = to_bgr(pil_img)
    orig_h, orig_w = bgr_original.shape[:2]

    if save_debug:
        p = _save_debug(bgr_original, "01_original", debug_tag)
        if p:
            debug_paths.append(p)

    # Step 2: Resize
    bgr_resized = resize_for_ocr(bgr_original, max_side=max_side)

    if save_debug:
        p = _save_debug(bgr_resized, "02_resized", debug_tag)
        if p:
            debug_paths.append(p)

    # Step 3: Enhance
    bgr_enhanced = enhance_for_ocr(bgr_resized)

    if save_debug:
        p = _save_debug(bgr_enhanced, "03_enhanced", debug_tag)
        if p:
            debug_paths.append(p)

    proc_h, proc_w = bgr_enhanced.shape[:2]
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "original_wh": (orig_w, orig_h),
        "processed_wh": (proc_w, proc_h),
        "processed": bgr_enhanced,
        "preprocessing_ms": elapsed_ms,
        "debug_paths": debug_paths,
    }
