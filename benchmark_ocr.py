"""
benchmark_ocr.py — EasyOCR vs PaddleOCR-VL benchmark
======================================================

Tests the EasyOCR experimental scanner on all available ID card images.
Does NOT require the FastAPI server to be running.

Usage
-----
    .venv\\Scripts\\python.exe benchmark_ocr.py
    .venv\\Scripts\\python.exe benchmark_ocr.py --expected "id.jpeg=DENNIS SABU"
    .venv\\Scripts\\python.exe benchmark_ocr.py --rotations
    .venv\\Scripts\\python.exe benchmark_ocr.py --preproc-compare
    set SAVE_DEBUG_IMAGES=1 && .venv\\Scripts\\python.exe benchmark_ocr.py

Options
-------
--images PATH ...
    Image paths to test. Default: auto-discover *.jpg *.jpeg *.png in project root.

--expected FILE=NAME ...
    Expected names, e.g.  --expected "id.jpeg=DENNIS SABU"

--rotations
    Also test each image at 90, 180, 270 degrees to verify orientation handling.

--preproc-compare
    Compare three preprocessing variants:
    A) Resize only
    B) CLAHE enhanced
    C) CLAHE + sharpening  (default)

--warm-runs N
    Number of warm runs per image (default: 1). Use 2+ to see stable warm timing.
"""

from __future__ import annotations

import argparse
import io as _io
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image as PILImage

# UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).parent
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EasyOCR ID card benchmark")
    p.add_argument("--images", nargs="+", metavar="PATH")
    p.add_argument("--expected", nargs="+", metavar="FILE=NAME", default=[])
    p.add_argument("--rotations", action="store_true")
    p.add_argument("--preproc-compare", action="store_true")
    p.add_argument("--warm-runs", type=int, default=1)
    return p.parse_args()


def discover_images(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    found = []
    for pat in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        found.extend(root.glob(pat))
    return sorted(set(found))


def parse_expected(items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items:
        if "=" in item:
            fname, _, name = item.partition("=")
            result[fname.strip()] = name.strip()
    return result


# ---------------------------------------------------------------------------
# Preprocessing variants
# ---------------------------------------------------------------------------


def _img_to_bgr(image_bytes: bytes) -> np.ndarray:
    pil = PILImage.open(_io.BytesIO(image_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(pil, dtype=np.uint8), cv2.COLOR_RGB2BGR)


def _resize(bgr: np.ndarray, max_side: int = 1280) -> np.ndarray:
    h, w = bgr.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return bgr
    scale = max_side / longest
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)


def _clahe(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge([cl, a, b]), cv2.COLOR_LAB2BGR)


def _sharpen(bgr: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(bgr, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(bgr, 1.5, blur, -0.5, 0)


def preprocess_A(image_bytes: bytes) -> np.ndarray:
    """Variant A: resize only."""
    return _resize(_img_to_bgr(image_bytes))


def preprocess_B(image_bytes: bytes) -> np.ndarray:
    """Variant B: resize + CLAHE."""
    return _clahe(_resize(_img_to_bgr(image_bytes)))


def preprocess_C(image_bytes: bytes) -> np.ndarray:
    """Variant C: resize + CLAHE + unsharp mask (default)."""
    return _sharpen(_clahe(_resize(_img_to_bgr(image_bytes))))


PREPROC_VARIANTS = {
    "A_resize_only":     preprocess_A,
    "B_clahe":           preprocess_B,
    "C_clahe_sharpen":   preprocess_C,
}


# ---------------------------------------------------------------------------
# Rotate image bytes
# ---------------------------------------------------------------------------


def rotate_bytes(image_bytes: bytes, angle: int) -> bytes:
    bgr = _img_to_bgr(image_bytes)
    rot_k = {90: 3, 180: 2, 270: 1}.get(angle, 0)
    bgr_rot = np.rot90(bgr, k=rot_k)
    _, enc = cv2.imencode(".jpg", bgr_rot, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return enc.tobytes()


# ---------------------------------------------------------------------------
# Single scan helper
# ---------------------------------------------------------------------------


def run_scan(scanner, image_bytes: bytes) -> dict:
    return scanner.scan_image_bytes(image_bytes)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------


def benchmark_images(
    scanner,
    images: list[Path],
    expected: dict[str, str],
    warm_runs: int,
    test_rotations: bool,
) -> list[dict]:
    """Run scanner on each image, collect results."""
    results = []
    angles_to_test = [0, 90, 180, 270] if test_rotations else [0]

    for img_path in images:
        raw_bytes = img_path.read_bytes()
        exp_name = expected.get(img_path.name, "")

        for angle in angles_to_test:
            label = img_path.name if angle == 0 else f"{img_path.name}@{angle}deg"
            test_bytes = raw_bytes if angle == 0 else rotate_bytes(raw_bytes, angle)

            print(f"\n{'-' * 60}")
            print(f"  Image: {label}")
            if exp_name:
                print(f"  Expected: {exp_name}")
            print(f"{'-' * 60}")

            # First run (may include model load if first ever request)
            r = run_scan(scanner, test_bytes)
            detected = r.get("name") or ""
            timing = r.get("timing", {})

            # Warm runs
            warm_times = []
            for _ in range(warm_runs - 1):
                wr = run_scan(scanner, test_bytes)
                warm_times.append(wr.get("timing", {}).get("total_ms", 0))

            avg_warm = int(sum(warm_times) / len(warm_times)) if warm_times else timing.get("total_ms", 0)

            correct: bool | None = None
            if exp_name:
                correct = detected.strip().upper() == exp_name.strip().upper()

            results.append({
                "image": label,
                "expected": exp_name,
                "detected": detected,
                "success": r.get("success", False),
                "correct": correct,
                "ocr_confidence": r.get("ocr_confidence", 0.0),
                "candidate_score": r.get("candidate_score", 0.0),
                "raw_text": r.get("raw_text", []),
                "preprocessing_ms": timing.get("preprocessing_ms", 0),
                "ocr_ms": timing.get("ocr_ms", 0),
                "name_extraction_ms": timing.get("name_extraction_ms", 0),
                "total_ms": timing.get("total_ms", 0),
                "avg_warm_ms": avg_warm,
            })

    return results


# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------


def print_results(results: list[dict], title: str) -> None:
    SEP = "=" * 72
    print()
    print(SEP)
    print(f"  {title}")
    print(SEP)

    for r in results:
        print()
        print(f"  Image:       {r['image']}")
        if r["expected"]:
            tag = "[YES]" if r["correct"] else "[NO] "
            print(f"  Expected:    {r['expected']}")
            print(f"  Detected:    {r['detected'] or '(none)'}")
            print(f"  Correct:     {tag}")
        else:
            print(f"  Detected:    {r['detected'] or '(none)'}")
        print(f"  OCR conf:    {r['ocr_confidence']:.3f}")
        print(f"  Name score:  {r['candidate_score']:.3f}")
        print()
        print(f"  Preprocessing:  {r['preprocessing_ms']:>5} ms")
        print(f"  OCR inference:  {r['ocr_ms']:>5} ms")
        print(f"  Name extract:   {r['name_extraction_ms']:>5} ms")
        print(f"  TOTAL (run 1):  {r['total_ms']:>5} ms")
        if r["avg_warm_ms"] != r["total_ms"]:
            print(f"  Avg warm:       {r['avg_warm_ms']:>5} ms")
        print()
        print(f"  Raw OCR ({len(r['raw_text'])} lines):")
        for line in r["raw_text"]:
            print(f"    {line!r}")
        print(f"  {'-' * 68}")

    # Summary
    print()
    print(SEP)
    print("  SUMMARY")
    print(SEP)
    if results:
        totals = [r["total_ms"] for r in results]
        warms  = [r["avg_warm_ms"] for r in results]
        ocrs   = [r["ocr_ms"] for r in results]
        print(f"  Images tested:     {len(results)}")
        print(f"  Avg total time:    {sum(totals)//len(totals)} ms")
        print(f"  Avg warm time:     {sum(warms)//len(warms)} ms")
        print(f"  Avg OCR time:      {sum(ocrs)//len(ocrs)} ms")
        print(f"  Min / Max total:   {min(totals)} ms / {max(totals)} ms")
        with_expected = [r for r in results if r["expected"]]
        if with_expected:
            correct = sum(1 for r in with_expected if r["correct"])
            pct = 100 * correct / len(with_expected)
            print(f"  Accuracy:          {correct}/{len(with_expected)} ({pct:.1f}%)")
        detected = sum(1 for r in results if r["success"])
        print(f"  Names detected:    {detected}/{len(results)} ({100*detected/len(results):.1f}%)")
    print(SEP)
    print()


# ---------------------------------------------------------------------------
# Preprocessing variant comparison
# ---------------------------------------------------------------------------


def compare_preprocessing(reader, images: list[Path], expected: dict[str, str]) -> None:
    print()
    print("=" * 72)
    print("  PREPROCESSING VARIANT COMPARISON")
    print("=" * 72)

    for img_path in images:
        raw_bytes = img_path.read_bytes()
        exp_name = expected.get(img_path.name, "")

        print(f"\n  Image: {img_path.name}  (expected: {exp_name or 'unknown'})")
        print(f"  {'-' * 60}")
        print(f"  {'Variant':<20} {'Detected':<25} {'OCR ms':>7}  {'Texts'}")
        print(f"  {'-' * 60}")

        for variant_name, prep_fn in PREPROC_VARIANTS.items():
            bgr = prep_fn(raw_bytes)

            t_ocr = time.perf_counter()
            try:
                raw_result = reader.readtext(bgr, detail=1, paragraph=False)
            except Exception as exc:
                print(f"  {variant_name:<20} ERROR: {exc}")
                continue
            ocr_ms = int((time.perf_counter() - t_ocr) * 1000)

            from app.scanner_ocr import (
                MIN_OCR_CONFIDENCE,
                _is_valid_name_candidate,
                _apply_vip_correction,
                _score_name_candidate,
            )
            lines = [
                {"text": str(t).strip(), "confidence": float(c)}
                for (_, t, c) in raw_result
                if str(t).strip() and float(c) >= MIN_OCR_CONFIDENCE
            ]

            raw_texts = [ln["text"] for ln in lines]
            candidates = []
            for item in lines:
                if _is_valid_name_candidate(item["text"]):
                    corr, _ = _apply_vip_correction(item["text"])
                    ns = _score_name_candidate(corr)
                    candidates.append({"text": corr, "blended": 0.5*item["confidence"] + 0.5*ns})

            detected = ""
            if candidates:
                candidates.sort(key=lambda c: c["blended"], reverse=True)
                detected = candidates[0]["text"]

            correct_tag = ""
            if exp_name:
                ok = detected.strip().upper() == exp_name.strip().upper()
                correct_tag = " [YES]" if ok else " [NO]"

            print(
                f"  {variant_name:<20} {(detected or '(none)'):<25}{correct_tag}  "
                f"{ocr_ms:>5} ms   {raw_texts}"
            )

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    if args.images:
        images = [Path(p) for p in args.images]
        images = [p for p in images if p.exists()]
        if not images:
            print("No valid image files found for the specified --images paths.")
            sys.exit(1)
    else:
        images = discover_images(TEST_IMAGES_DIR)
        if not images:
            print(f"Error: No test images (.jpg, .jpeg, .png, .webp) found in '{TEST_IMAGES_DIR}'.")
            sys.exit(1)

    expected = parse_expected(args.expected)

    print()
    print("=" * 72)
    print("  ASTHRA ID SCANNER — EasyOCR Benchmark")
    print("=" * 72)
    for p in images:
        print(f"  {p.name}  ({p.stat().st_size / 1024:.1f} KB)")
    print()

    # ---- Load EasyOCR ----
    from app.scanner_ocr import LightOCRScanner

    scanner = LightOCRScanner()

    t_load = time.perf_counter()
    if not scanner._ensure_loaded():
        print(f"[ERROR] EasyOCR failed to load: {scanner.load_error}")
        sys.exit(1)
    load_time = time.perf_counter() - t_load
    print(f"  Model load time: {load_time:.1f}s")
    print()

    # ---- Preprocessing comparison ----
    if args.preproc_compare:
        compare_preprocessing(scanner._reader, images, expected)

    # ---- Main benchmark ----
    results = benchmark_images(
        scanner, images, expected,
        warm_runs=args.warm_runs,
        test_rotations=args.rotations,
    )
    print_results(results, "EASYOCR BENCHMARK RESULTS")


if __name__ == "__main__":
    main()
