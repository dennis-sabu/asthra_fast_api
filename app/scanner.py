"""
scanner.py — Asthra ID Card Name Scanner
=========================================

Architecture:
    PaddleOCR-VL 1.6 (0.9B) is the only working OCR pipeline on this system.
    (Standard PP-OCRv6 fails with a PaddlePaddle oneDNN incompatibility.)

    The model is loaded ONCE at startup with all unnecessary sub-pipelines
    disabled (orientation classify, unwarping, chart, seal). This is the
    primary speed optimization.

    Per-request optimizations:
        - max_new_tokens=64 (name needs <30 tokens; default is uncapped)
        - max_pixels reduced (smaller image region fed to VLM = faster)
        - Image preprocessed with CLAHE for better text contrast
        - Images resized to a sane max side before inference
        - Duplicate-frame detection via MD5 hash cache (avoids re-scanning
          the same frame multiple times when frontend sends burst requests)

    Name extraction:
        - All text blocks from VL output are scored as name candidates
        - VIP correction layer: if an OCR candidate is close enough to a
          known important name, the canonical form is returned
        - Validation rejects obvious non-names (IDs, course codes, etc.)
        - If no confident name is found → success=False, name=None

VIP list:
    Used ONLY for candidate correction — never to block unknown visitors.
    Any visitor's name found on the card is returned correctly.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from difflib import SequenceMatcher
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Disable model-source connectivity check (models already cached locally)
# ---------------------------------------------------------------------------
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# ---------------------------------------------------------------------------
# VIP name list — correction/validation only, NOT a whitelist
# ---------------------------------------------------------------------------

VIP_NAMES: list[str] = [
    "Msgr. Dr. Joseph Thadathil",
    "Rev. Prof. Dr. James John Mangalathu",
    "Dr. V. P. Devassia",
    "Rev. Dr. Joseph Purayidathil",
    "Dr. Giby Jose",
]

# Minimum similarity to accept a VIP correction
VIP_SIMILARITY_THRESHOLD = 0.75

# Image preprocessing: max side length before passing to VLM (pixels)
MAX_IMAGE_SIDE = 1280

# Max upload bytes allowed: 10 MB
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# VL inference: limit output tokens — a person's name needs <30 tokens
VL_MAX_NEW_TOKENS = 64

# VL inference: limit image pixels fed to the VLM (28*28 = one ViT patch)
# Benchmark results (CPU, RTX 3050 not available via this paddle build):
#   256 * 28 * 28 ≈ 200K px → ~20s  (Dennis Sabu: ✓)
#   128 * 28 * 28 ≈ 100K px → ~18s  (Dennis Sabu: ✓)  (previous default)
#    64 * 28 * 28 ≈  50K px → ~18s  (Dennis Sabu: ✓)  (diminishing returns)
# The above was only ever benchmarked against ONE card, and 100K->200K cost
# just +2s for the same result, so headroom above 256 was never explored.
# At 128 (~400x250 effective px for a 1.585:1 card) small/lighter-weight
# printed names can fall below legible resolution, which likely explains
# reliability varying across different cards rather than a flat pass/fail.
# 512 (~798x503 effective px) roughly doubles linear resolution for what
# the 100K->200K step suggests should be a modest time cost. This is a
# reversible experiment: re-run the repeated-scan test protocol at this
# setting and compare success rate + processing_time_ms against the
# previous VL_MAX_PIXELS = 128 * 28 * 28 baseline before keeping it.
VL_MAX_PIXELS = 512 * 28 * 28
VL_MIN_PIXELS = 4 * 28 * 28

# ---------------------------------------------------------------------------
# Non-name keyword reject list (uppercased for comparison)
# ---------------------------------------------------------------------------

_REJECT_WORDS: frozenset[str] = frozenset({
    "ST.JOSEPH", "ST. JOSEPH", "COLLEGE", "ENGINEERING",
    "TECHNOLOGY", "AUTONOMOUS", "B.TECH", "BTECH", "M.TECH", "MTECH",
    "ECS", "ECE", "CSE", "COMPUTER", "ELECTRONICS", "MECHANICAL",
    "PRINCIPAL", "SIGNATURE", "MANAGED", "DIOCESE", "DEPARTMENT",
    "UNIVERSITY", "INSTITUTION", "ACADEMY", "SCHOOL",
    "IDENTIFICATION", "VALID", "VALIDITY", "ISSUED", "ISSUE",
    "EXPIRY", "EXPIRES", "ADDRESS", "PHONE", "EMAIL",
    "WEBSITE", "WWW.", "HTTP", "EMPLOYEE", "STUDENT",
})

# Regex: numeric IDs / registration numbers
_ID_PATTERN = re.compile(r"^\d+$|^[A-Z]{1,4}\d{3,}$")

# Regex: date strings
_DATE_PATTERN = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    r"|\b\d{4}[/-]\d{2}[/-]\d{2}\b"
)

# Honorific tokens (used for scoring and some validation exceptions)
_HONORIFICS: frozenset[str] = frozenset({
    "Dr", "Dr.", "Rev", "Rev.", "Prof", "Prof.",
    "Msgr", "Msgr.", "Mr", "Mr.", "Mrs", "Mrs.", "Ms", "Ms.",
    "Er", "Er.", "Fr", "Fr.",
})


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def image_hash(data: bytes) -> str:
    """MD5 digest of raw image bytes — used for duplicate-frame caching."""
    return hashlib.md5(data).hexdigest()


def _similarity(a: str, b: str) -> float:
    """Case-insensitive character-level similarity ratio."""
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def apply_vip_correction(candidate: str) -> tuple[str, bool]:
    """
    Return (corrected_name, was_corrected).

    If `candidate` is sufficiently close to a VIP name, return that VIP name.
    Otherwise return the original candidate unchanged.
    """
    best_score = 0.0
    best_vip = candidate

    for vip in VIP_NAMES:
        score = _similarity(candidate, vip)
        if score > best_score:
            best_score = score
            best_vip = vip

    if best_score >= VIP_SIMILARITY_THRESHOLD:
        return best_vip, True

    return candidate, False


def is_valid_name_candidate(text: str) -> bool:
    """
    Conservative name validator.

    Returns True if `text` could plausibly be a person's name.
    Allows honorifics, initials, hyphens, apostrophes.
    Rejects IDs, institution names, course codes, dates, pure symbols.
    """
    text = text.strip()
    if not text:
        return False

    upper = text.upper()

    # Reject known non-name keywords
    if any(w in upper for w in _REJECT_WORDS):
        return False

    # Reject pure ID patterns (e.g. "24ES031")
    if _ID_PATTERN.match(text):
        return False

    # Reject text containing dates
    if _DATE_PATTERN.search(text):
        return False

    # Must contain at least 2 letters
    if sum(c.isalpha() for c in text) < 2:
        return False

    # Reject if >15% digits (likely a code)
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    if digit_ratio > 0.15:
        return False

    # Allow only name-safe characters
    if not all(c.isalpha() or c in " .'-," for c in text):
        return False

    words = text.split()

    # Minimum 2 words
    if len(words) < 2:
        return False

    # Maximum 8 words (longer = likely a label/sentence)
    if len(words) > 8:
        return False

    # Reject ALL-UPPER lines with 3+ non-honorific words (likely a header)
    non_honorific = [w for w in words if w.rstrip(".").title() not in _HONORIFICS]
    if len(non_honorific) >= 3 and all(
        w.isupper() for w in non_honorific if len(w) > 1
    ):
        return False

    return True


def score_name_candidate(text: str) -> float:
    """
    Heuristic quality score for a name candidate (0.0–1.0).
    Higher = more likely to be a real person's name.
    """
    words = text.split()
    score = 0.0

    # Word count sweet spot: 2–4
    if 2 <= len(words) <= 4:
        score += 0.30
    elif len(words) == 5:
        score += 0.15

    # Total length sweet spot
    if 6 <= len(text) <= 45:
        score += 0.20

    # Starts with a recognised honorific
    first = words[0].rstrip(".")
    if first in {h.rstrip(".") for h in _HONORIFICS}:
        score += 0.30

    # Most words are title-cased (typical for names)
    cap_count = sum(1 for w in words if w and w[0].isupper())
    if cap_count >= max(1, len(words) - 1):
        score += 0.20

    # VIP proximity bonus
    for vip in VIP_NAMES:
        if _similarity(text, vip) > 0.80:
            score += 0.20
            break

    return min(score, 1.0)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode image bytes → BGR numpy array, resized + CLAHE-enhanced.

    CLAHE improves text contrast on ID cards with uneven lighting,
    which helps the VLM read faint text more reliably.
    """
    pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = pil_img.size

    # Resize if too large (preserving aspect ratio)
    max_side = max(orig_w, orig_h)
    if max_side > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max_side
        pil_img = pil_img.resize(
            (int(orig_w * scale), int(orig_h * scale)),
            Image.LANCZOS,
        )

    bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # CLAHE contrast enhancement on L channel
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.merge([clahe.apply(l_ch), a_ch, b_ch])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# IDScanner
# ---------------------------------------------------------------------------


class IDScanner:
    """
    Manages the PaddleOCR-VL 1.6 pipeline for ID card name extraction.

    The pipeline is loaded ONCE at startup with all unnecessary sub-modules
    disabled.  Every call to `scan_image_bytes` reuses this loaded pipeline.

    Key optimizations applied at inference time:
        - max_new_tokens=64  (VLM stops after name is generated)
        - max_pixels reduced  (smaller image patch = fewer transformer ops)
        - Duplicate-frame MD5 cache (same frame never processed twice)
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._model_loaded = False
        self._device = "cpu"
        self._cache: dict[str, dict] = {}  # {md5: result_dict}
        self._cache_max = 30

        self._load_pipeline()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def device(self) -> str:
        return self._device

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_pipeline(self) -> None:
        t0 = time.time()

        print()
        print("=" * 56)
        print(" Loading PaddleOCR-VL 1.6...")
        print("=" * 56)

        try:
            from paddleocr import PaddleOCRVL

            self._pipeline = PaddleOCRVL(
                pipeline_version="v1.6",
                # Disable every sub-module not needed for a flat ID card
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_chart_recognition=False,
                use_seal_recognition=False,
                # Keep layout detection: needed to find text blocks
                use_layout_detection=True,
                # Don't convert to Markdown — we parse raw blocks
                format_block_content=False,
                merge_layout_blocks=True,
                # Disable async queue — we process one image at a time
                use_queues=False,
            )

            elapsed = time.time() - t0
            self._model_loaded = True
            print(f" PaddleOCR-VL loaded in {elapsed:.1f}s  |  device={self._device}")

        except Exception as exc:
            print(f" ERROR loading PaddleOCR-VL: {exc}")
            self._model_loaded = False

        print("=" * 56)
        print()

    # ------------------------------------------------------------------
    # Main public entry point
    # ------------------------------------------------------------------

    def scan_image_bytes(self, image_bytes: bytes) -> dict:
        """
        Process raw image bytes and return a scan result dict.

        Return schema:
            {
                "success": bool,
                "name": str | None,
                "message": str,
                "confidence": float,
                "processing_time_ms": int,
            }
        """
        t_start = time.time()

        if not self._model_loaded or self._pipeline is None:
            return self._failure("Model not loaded.", t_start)

        # ---- Duplicate-frame cache ----
        img_hash = image_hash(image_bytes)
        if img_hash in self._cache:
            cached = dict(self._cache[img_hash])
            cached["processing_time_ms"] = int((time.time() - t_start) * 1000)
            print(f"[Scanner] Cache hit -- returning cached result instantly")
            return cached

        # ---- Preprocess ----
        try:
            bgr = preprocess_image(image_bytes)
        except Exception as exc:
            return self._failure(f"Image decode failed: {exc}", t_start)

        h, w = bgr.shape[:2]
        print(f"[Scanner] Processing {w}x{h} image...")

        # ---- Run VL inference ----
        result = self._run_vl(bgr, t_start)

        # ---- Cache successful results ----
        if result["success"]:
            self._store_cache(img_hash, result)

        return result

    # ------------------------------------------------------------------
    # VL inference
    # ------------------------------------------------------------------

    def _run_vl(self, bgr: np.ndarray, t_start: float) -> dict:
        import tempfile

        # Write preprocessed image to a temp file (VL pipeline needs a path)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])

        t_infer = time.time()

        try:
            raw = self._pipeline.predict(
                tmp_path,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_chart_recognition=False,
                use_seal_recognition=False,
                # Key speed params: limit VLM output and image patch count
                max_new_tokens=VL_MAX_NEW_TOKENS,
                max_pixels=VL_MAX_PIXELS,
                min_pixels=VL_MIN_PIXELS,
            )
        except Exception as exc:
            print(f"[Scanner] VL inference error: {exc}")
            return self._failure(f"Inference error: {exc}", t_start)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        infer_ms = int((time.time() - t_infer) * 1000)
        print(f"[Scanner] VL inference: {infer_ms}ms")

        texts = self._parse_vl_output(raw)
        return self._pick_best_name(texts, t_start)

    def _parse_vl_output(self, raw_results) -> list[dict]:
        """Extract text blocks from PaddleOCR-VL output."""
        texts: list[dict] = []

        for result in raw_results:
            try:
                data = result.json
                if callable(data):
                    data = data()

                res = data.get("res", data)
                blocks = res.get("parsing_res_list", [])

                for block in blocks:
                    label = block.get("block_label", "")
                    if label in ("text", "paragraph_title"):
                        content = block.get("block_content", "").strip()
                        if content:
                            texts.append({"text": content, "score": 0.85})
            except Exception:
                pass

        # -- DEBUG: print raw OCR text blocks --------------------------
        print()
        print("=" * 34 + " RAW OCR TEXT " + "=" * 34)
        if texts:
            for t in texts:
                print(t["text"])
        else:
            print("(no text/paragraph_title blocks found)")
        print("=" * 82)
        print()
        # --------------------------------------------------------------

        return texts

    # ------------------------------------------------------------------
    # Name extraction
    # ------------------------------------------------------------------

    def _pick_best_name(self, texts: list[dict], t_start: float) -> dict:
        """
        Score all text candidates and pick the best person's name.
        Applies VIP correction before scoring.
        """
        candidates: list[dict] = []

        for item in texts:
            raw_text = item["text"].strip()
            ocr_score = float(item.get("score", 0.85))

            # Split multi-line blocks into individual lines
            lines = [
                ln.strip()
                for ln in re.split(r"[\n|]", raw_text)
                if ln.strip()
            ]

            for line in lines:
                if not is_valid_name_candidate(line):
                    print(f"[Scanner] REJECTED: {line!r}")
                    continue

                corrected, was_corrected = apply_vip_correction(line)
                name_score = score_name_candidate(corrected)

                # Blend OCR confidence with name-quality heuristic
                confidence = 0.5 * ocr_score + 0.5 * name_score

                candidates.append({
                    "text": corrected,
                    "original": line,
                    "was_corrected": was_corrected,
                    "confidence": confidence,
                })

        elapsed_ms = int((time.time() - t_start) * 1000)

        if not candidates:
            print("[Scanner] No name candidate found.")
            return {
                "success": False,
                "name": None,
                "message": "Couldn't read the name clearly. Please try again.",
                "confidence": 0.0,
                "processing_time_ms": elapsed_ms,
            }

        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        winner = candidates[0]

        name = winner["text"]
        confidence = round(winner["confidence"], 3)
        correction = (
            f" [VIP-corrected from {winner['original']!r}]"
            if winner["was_corrected"] and winner["original"] != name
            else ""
        )

        print(
            f"[Scanner] Name: {name!r}{correction} | "
            f"conf={confidence:.2f} | {elapsed_ms}ms"
        )

        return {
            "success": True,
            "name": name,
            "message": f"Welcome, {name}!",
            "confidence": confidence,
            "processing_time_ms": elapsed_ms,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _failure(self, reason: str, t_start: float) -> dict:
        elapsed_ms = int((time.time() - t_start) * 1000)
        print(f"[Scanner] FAILURE: {reason}")
        return {
            "success": False,
            "name": None,
            "message": "Couldn't read the name clearly. Please try again.",
            "confidence": 0.0,
            "processing_time_ms": elapsed_ms,
        }

    def _store_cache(self, img_hash: str, result: dict) -> None:
        """Cache a result, evicting the oldest entry when full."""
        if len(self._cache) >= self._cache_max:
            self._cache.pop(next(iter(self._cache)))
        self._cache[img_hash] = result