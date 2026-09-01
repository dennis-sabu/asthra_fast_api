"""
scanner_ocr.py — Experimental EasyOCR ID Card Scanner
=======================================================

EXPERIMENTAL scanner on the `paddleocr-test` branch.
Runs alongside the existing PaddleOCR-VL scanner (scanner.py).
Does NOT replace it.

Architecture
------------
    FastAPI startup
        (nothing — lazy loading)
        ↓ on first /test-paddle-ocr request
    easyocr.Reader(["en"], gpu=False)
        ↓
    OpenCV preprocessing  (preprocessing.py)
        ↓
    Smart orientation: run once, rotate only if quality is poor
        ↓
    EasyOCR inference  → list of (bbox, text, confidence)
        ↓
    Name candidate scoring
        ↓
    Return result with full timing breakdown

Why EasyOCR instead of PaddleOCR standard?
--------------------------------------------
PaddlePaddle 3.3.1 has a confirmed binary-level incompatibility with the
PIR/oneDNN executor on this machine.  All standard CNN detection models
(PP-OCRv4/v5/v6) crash with:
    NotImplementedError: ConvertPirAttribute2RuntimeAttribute
EasyOCR uses PyTorch and avoids this entirely.

Preprocessing variants tested per scan (configurable)
------------------------------------------------------
A: Resize only              (min processing, max text preservation)
B: CLAHE enhanced           (better contrast on uneven lighting)
C: CLAHE + unsharp mask     (sharper text edges)

The scanner uses variant C by default.  The benchmark script compares all
three to find the best variant for the real ID cards.

Orientation handling
---------------------
Run OCR on original image first.
Evaluate quality: confidence + word count + alpha chars + name candidates.
Only try rotations when quality score is below threshold — avoids 4× cost
on normally-oriented cards.
"""

from __future__ import annotations

import os
import re
import sys
import time
from difflib import SequenceMatcher
from typing import Optional

import cv2
import numpy as np

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from .preprocessing import preprocess_pipeline, rotate_image, enhance_for_ocr, resize_for_ocr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_UPLOAD_BYTES = 10 * 1024 * 1024   # 10 MB — same as existing scanner

MAX_IMAGE_SIDE = 1280                  # Resize limit before OCR

# Minimum EasyOCR text confidence to accept (0.0–1.0)
MIN_OCR_CONFIDENCE = 0.3

# Orientation: quality score below this triggers rotation attempts.
# Threshold is deliberately generous (0.55) so that sideways cards producing
# high-confidence single characters (e.g. 'i', '1', 'g') still trigger rotation.
ORIENTATION_QUALITY_THRESHOLD = 0.55

# EasyOCR language list — English only to keep model small and fast
EASYOCR_LANGUAGES = ["en"]

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

VIP_SIMILARITY_THRESHOLD = 0.75

# ---------------------------------------------------------------------------
# Non-name keyword reject list
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
    "REGISTER", "ROLL", "REG.", "PHOTO", "LIBRARY", "ACCESS", "CARD",
    "PALAI", "DIOCESE", "THRISSUR", "KERALA", "INDIA",
})

_ID_PATTERN = re.compile(r"^\d+$|^[A-Z]{1,4}\d{3,}$|^\d{2,}[A-Z]{2,}\d{3,}$")
_DATE_PATTERN = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{2}[/-]\d{2}\b"
)
_HONORIFICS: frozenset[str] = frozenset({
    "Dr", "Dr.", "Rev", "Rev.", "Prof", "Prof.",
    "Msgr", "Msgr.", "Mr", "Mr.", "Mrs", "Mrs.", "Ms", "Ms.",
    "Er", "Er.", "Fr", "Fr.",
})


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio()


def _apply_vip_correction(candidate: str) -> tuple[str, bool]:
    best_score, best_vip = 0.0, candidate
    for vip in VIP_NAMES:
        s = _similarity(candidate, vip)
        if s > best_score:
            best_score, best_vip = s, vip
    if best_score >= VIP_SIMILARITY_THRESHOLD:
        return best_vip, True
    return candidate, False


def _is_valid_name_candidate(text: str) -> bool:
    """Return True if text could plausibly be a person's name."""
    text = text.strip()
    if not text:
        return False
    upper = text.upper()
    if any(w in upper for w in _REJECT_WORDS):
        return False
    if _ID_PATTERN.match(text.strip()):
        return False
    if _DATE_PATTERN.search(text):
        return False
    if sum(c.isalpha() for c in text) < 4:
        return False
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    if digit_ratio > 0.15:
        return False
    if not all(c.isalpha() or c in " .'-," for c in text):
        return False
    words = text.split()
    if len(words) < 2 or len(words) > 8:
        return False
    # Reject ALL-CAPS lines with 3+ non-honorific words (likely a heading)
    non_hon = [w for w in words if w.rstrip(".").title() not in _HONORIFICS]
    if len(non_hon) >= 3 and all(w.isupper() for w in non_hon if len(w) > 1):
        return False
    return True


def _score_name_candidate(text: str) -> float:
    """Heuristic quality score (0.0–1.0). Higher = more name-like."""
    words = text.split()
    score = 0.0
    if 2 <= len(words) <= 4:
        score += 0.30
    elif len(words) == 5:
        score += 0.15
    if 6 <= len(text) <= 45:
        score += 0.20
    first = words[0].rstrip(".")
    if first in {h.rstrip(".") for h in _HONORIFICS}:
        score += 0.30
    cap_count = sum(1 for w in words if w and w[0].isupper())
    if cap_count >= max(1, len(words) - 1):
        score += 0.20
    for vip in VIP_NAMES:
        if _similarity(text, vip) > 0.80:
            score += 0.20
            break
    return min(score, 1.0)


def _orientation_quality(ocr_lines: list[dict]) -> float:
    """
    Score the overall OCR quality for orientation selection (0.0–1.0).

    Combines:
    - Average confidence of high-confidence lines
    - Number of alphabetic characters
    - Number of multi-word text segments (likely real words)
    - Presence of at least one valid name candidate
    """
    if not ocr_lines:
        return 0.0

    # Average confidence (weighted by text length)
    total_chars = sum(len(ln["text"]) for ln in ocr_lines)
    if total_chars == 0:
        return 0.0

    weighted_conf = sum(
        ln["confidence"] * len(ln["text"]) for ln in ocr_lines
    ) / total_chars

    # Alphabetic character ratio
    all_text = " ".join(ln["text"] for ln in ocr_lines)
    alpha_ratio = sum(c.isalpha() for c in all_text) / max(len(all_text), 1)

    # Multi-word segments
    multi_word = sum(1 for ln in ocr_lines if len(ln["text"].split()) >= 2)
    word_bonus = min(multi_word / 5.0, 0.2)

    # Name candidate bonus
    name_bonus = 0.1 if any(
        _is_valid_name_candidate(ln["text"]) for ln in ocr_lines
    ) else 0.0

    quality = (
        0.5 * weighted_conf
        + 0.2 * alpha_ratio
        + word_bonus
        + name_bonus
    )
    return min(quality, 1.0)


# ---------------------------------------------------------------------------
# LightOCRScanner (EasyOCR backend)
# ---------------------------------------------------------------------------


class LightOCRScanner:
    """
    Experimental ID card scanner using EasyOCR (English, CPU).

    Loaded LAZILY — model initialises on the first /test-paddle-ocr request.
    Subsequent scans reuse the loaded Reader.
    """

    def __init__(self) -> None:
        self._reader = None
        self._model_loaded = False
        self._load_error: Optional[str] = None

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    # ------------------------------------------------------------------
    # Lazy loader
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> bool:
        if self._model_loaded:
            return True
        if self._load_error is not None:
            return False

        print()
        print("=" * 56)
        print(" Loading EasyOCR (English, CPU)...")
        print("=" * 56)
        t0 = time.perf_counter()

        try:
            import easyocr  # noqa: PLC0415
            self._reader = easyocr.Reader(
                EASYOCR_LANGUAGES,
                gpu=False,
                verbose=False,
            )
            elapsed = time.perf_counter() - t0
            self._model_loaded = True
            print(f" EasyOCR loaded in {elapsed:.1f}s")
            print("=" * 56)
            print()
            return True

        except Exception as exc:
            self._load_error = str(exc)
            print(f" ERROR loading EasyOCR: {exc}")
            print("=" * 56)
            print()
            return False

    # ------------------------------------------------------------------
    # Main public entry point
    # ------------------------------------------------------------------

    def scan_image_bytes(self, image_bytes: bytes) -> dict:
        """
        Process raw image bytes, run EasyOCR, return structured result.

        Returns
        -------
        {
            "success": bool,
            "name": str | None,
            "message": str,
            "raw_text": list[str],
            "ocr_confidence": float,
            "candidate_score": float,
            "timing": {
                "preprocessing_ms": int,
                "ocr_ms": int,
                "name_extraction_ms": int,
                "total_ms": int,
            }
        }
        """
        t_total = time.perf_counter()

        print()
        print("=" * 56)
        print(" EASYOCR SCAN")
        print("=" * 56)

        if not self._ensure_loaded():
            return self._failure(
                f"Model failed to load: {self._load_error}",
                t_total, 0, 0,
            )

        # ---- Preprocess ----
        debug_on = os.environ.get("SAVE_DEBUG_IMAGES", "0") == "1"
        try:
            prep = preprocess_pipeline(
                image_bytes,
                max_side=MAX_IMAGE_SIDE,
                save_debug=debug_on,
                debug_tag=str(int(t_total)),
            )
        except ValueError as exc:
            return self._failure(str(exc), t_total, 0, 0)

        preproc_ms = prep["preprocessing_ms"]
        bgr = prep["processed"]
        orig_w, orig_h = prep["original_wh"]
        proc_w, proc_h = prep["processed_wh"]

        print(f" Input:    {orig_w}x{orig_h}  ->  processed: {proc_w}x{proc_h}")
        print(f" Preproc:  {preproc_ms} ms")

        # ---- OCR with smart orientation ----
        t_ocr = time.perf_counter()
        ocr_lines, best_angle, best_quality = self._run_ocr_with_orientation(bgr)
        ocr_ms = int((time.perf_counter() - t_ocr) * 1000)

        print(f" OCR:      {ocr_ms} ms  (angle={best_angle}  quality={best_quality:.2f})")
        print(f" Lines:    {len(ocr_lines)} detected")

        # ---- Name extraction ----
        t_name = time.perf_counter()
        in_sweep = best_angle != 0
        result = self._extract_name(ocr_lines, preproc_ms, ocr_ms, in_rotation_sweep=in_sweep)
        name_ms = int((time.perf_counter() - t_name) * 1000)
        total_ms = int((time.perf_counter() - t_total) * 1000)

        result["timing"]["name_extraction_ms"] = name_ms
        result["timing"]["total_ms"] = total_ms

        # ---- Summary log ----
        print()
        print("=" * 56)
        print(f" Result:   success={result['success']}  name={result.get('name')!r}")
        print(f" Preproc:  {preproc_ms} ms")
        print(f" OCR:      {ocr_ms} ms")
        print(f" Name ext: {name_ms} ms")
        print(f" TOTAL:    {total_ms} ms")
        print("=" * 56)
        print()

        return result

    # ------------------------------------------------------------------
    # OCR with smart orientation
    # ------------------------------------------------------------------

    def _run_ocr_with_orientation(
        self, bgr: np.ndarray
    ) -> tuple[list[dict], int, float]:
        """
        Run OCR, correcting orientation only when quality is poor.

        Strategy:
        1. Run OCR on original orientation.
        2. Evaluate quality: confidence + alpha chars + multi-word segments
           + name candidate presence.
        3. If quality >= threshold AND at least one name candidate found,
           accept original orientation.
        4. Otherwise try 180, 90, 270 (180 first — most common mistake).
           Stop early once quality exceeds threshold AND a name candidate exists.

        Returns (ocr_lines, best_angle, best_quality_score).
        """
        original_lines = self._ocr_single(bgr)
        orig_quality = _orientation_quality(original_lines)
        has_name = any(_is_valid_name_candidate(ln["text"]) for ln in original_lines)

        if orig_quality >= ORIENTATION_QUALITY_THRESHOLD and has_name:
            return original_lines, 0, orig_quality

        reason = (
            f"quality={orig_quality:.2f} < {ORIENTATION_QUALITY_THRESHOLD}"
            if orig_quality < ORIENTATION_QUALITY_THRESHOLD
            else "no name candidate found"
        )
        print(f" [Orient] {reason} — trying rotations...")

        best_lines, best_angle, best_quality = original_lines, 0, orig_quality
        best_has_name = has_name

        for angle in (180, 90, 270):   # 180 first — most common shooting mistake
            rotated = rotate_image(bgr, angle)
            lines = self._ocr_single(rotated)
            q = _orientation_quality(lines)
            found_name = any(_is_valid_name_candidate(ln["text"]) for ln in lines)
            print(f" [Orient] {angle:>3}deg: quality={q:.2f}  lines={len(lines)}  name={found_name}")

            # Prefer: name found > quality > previous best
            better = (
                (found_name and not best_has_name)
                or (found_name == best_has_name and q > best_quality)
            )
            if better:
                best_lines, best_angle, best_quality = lines, angle, q
                best_has_name = found_name

            if best_quality >= ORIENTATION_QUALITY_THRESHOLD and best_has_name:
                break  # Good enough — stop early

        return best_lines, best_angle, best_quality

    def _ocr_single(self, bgr: np.ndarray) -> list[dict]:
        """
        Run EasyOCR on one BGR image.
        Returns list of {'text': str, 'confidence': float, 'bbox': list}.
        """
        try:
            # EasyOCR expects RGB or BGR numpy array
            # detail=1 returns (bbox, text, confidence)
            raw = self._reader.readtext(bgr, detail=1, paragraph=False)
        except Exception as exc:
            print(f" [OCR] Inference error: {exc}")
            return []

        lines: list[dict] = []
        for (bbox, text, conf) in raw:
            text = str(text).strip()
            if text and float(conf) >= MIN_OCR_CONFIDENCE:
                lines.append({
                    "text": text,
                    "confidence": float(conf),
                    "bbox": bbox,
                })
        return lines

    # ------------------------------------------------------------------
    # Name extraction
    # ------------------------------------------------------------------

    def _extract_name(
        self,
        ocr_lines: list[dict],
        preproc_ms: int,
        ocr_ms: int,
        in_rotation_sweep: bool = False,
    ) -> dict:
        """Score all OCR lines and return the best name candidate.

        Parameters
        ----------
        in_rotation_sweep : bool
            When True (orientation was auto-corrected), apply a higher minimum
            OCR confidence (0.5) to prevent garbled text from passing as a name.
        """

        raw_text = [ln["text"] for ln in ocr_lines]

        # Print detected text for debugging
        print()
        print("=" * 28 + " RAW OCR TEXT " + "=" * 14)
        if raw_text:
            for ln in ocr_lines:
                print(f"  [{ln['confidence']:.3f}] {ln['text']!r}")
        else:
            print("  (no text detected)")
        print("=" * 56)
        print()

        timing = {
            "preprocessing_ms": preproc_ms,
            "ocr_ms": ocr_ms,
            "name_extraction_ms": 0,
            "total_ms": 0,
        }

        # Minimum OCR confidence for a name candidate.
        # Raised when a rotation sweep was needed to avoid low-confidence
        # reversed/garbled text being accepted as a person's name.
        min_name_conf = 0.50 if in_rotation_sweep else 0.30

        candidates: list[dict] = []

        for item in ocr_lines:
            raw = item["text"].strip()
            ocr_conf = float(item.get("confidence", 0.0))

            # Skip low-confidence lines during rotation sweep
            if ocr_conf < min_name_conf:
                print(f" [Name] SKIP (conf={ocr_conf:.3f} < {min_name_conf}): {raw!r}")
                continue

            # Split on newlines/pipes in case OCR joined lines
            for line in [s.strip() for s in re.split(r"[\n|]", raw) if s.strip()]:
                if not _is_valid_name_candidate(line):
                    print(f" [Name] REJECT: {line!r}")
                    continue
                corrected, was_vip = _apply_vip_correction(line)
                name_sc = _score_name_candidate(corrected)
                blended = 0.5 * ocr_conf + 0.5 * name_sc
                candidates.append({
                    "text": corrected,
                    "original": line,
                    "was_vip": was_vip,
                    "ocr_score": ocr_conf,
                    "name_score": name_sc,
                    "blended": blended,
                })

        if not candidates:
            print(" [Name] No valid name candidate found.")
            return {
                "success": False,
                "name": None,
                "message": "Couldn't identify a name clearly.",
                "raw_text": raw_text,
                "ocr_confidence": 0.0,
                "candidate_score": 0.0,
                "timing": timing,
            }

        candidates.sort(key=lambda c: c["blended"], reverse=True)
        winner = candidates[0]
        name = winner["text"]
        vip_note = (
            f" [VIP from {winner['original']!r}]"
            if winner["was_vip"] and winner["original"] != name
            else ""
        )
        print(
            f" [Name] Winner: {name!r}{vip_note}  "
            f"ocr={winner['ocr_score']:.3f}  "
            f"cand={winner['name_score']:.3f}  "
            f"blend={winner['blended']:.3f}"
        )

        return {
            "success": True,
            "name": name,
            "message": f"Welcome, {name}!",
            "raw_text": raw_text,
            "ocr_confidence": round(winner["ocr_score"], 3),
            "candidate_score": round(winner["name_score"], 3),
            "blended_score": round(winner["blended"], 3),
            "timing": timing,
        }

    # ------------------------------------------------------------------
    # Failure helper
    # ------------------------------------------------------------------

    def _failure(
        self,
        reason: str,
        t_total: float,
        preproc_ms: int,
        ocr_ms: int,
    ) -> dict:
        total_ms = int((time.perf_counter() - t_total) * 1000)
        print(f" [LightOCR] FAILURE: {reason}")
        return {
            "success": False,
            "name": None,
            "message": "Couldn't read the name clearly. Please try again.",
            "raw_text": [],
            "ocr_confidence": 0.0,
            "candidate_score": 0.0,
            "timing": {
                "preprocessing_ms": preproc_ms,
                "ocr_ms": ocr_ms,
                "name_extraction_ms": 0,
                "total_ms": total_ms,
            },
        }
