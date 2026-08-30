"""
benchmark_vl.py — Benchmark PaddleOCR-VL 1.6 with vs without optimizations

Run with:
    .venv\Scripts\python.exe benchmark_vl.py
"""

import time
import os

IMAGE_PATH = r"id.jpeg"

# Suppress model source check for speed (models already cached)
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

print("=" * 60)
print("ASTHRA OCR BENCHMARK — PaddleOCR-VL 1.6")
print("=" * 60)

print("\n[1/3] Loading PaddleOCR-VL 1.6 (optimized flags)...")
from paddleocr import PaddleOCRVL

t_load = time.time()

pipeline = PaddleOCRVL(
    pipeline_version="v1.6",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_chart_recognition=False,
    use_seal_recognition=False,
    format_block_content=False,
    merge_layout_blocks=True,
    use_queues=False,
)

load_time = time.time() - t_load
print(f"   Load time: {load_time:.1f}s")

print("\n[2/3] First scan (cold — JIT/model warm-up)...")
t1 = time.time()

output1 = pipeline.predict(
    IMAGE_PATH,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_chart_recognition=False,
    use_seal_recognition=False,
    max_new_tokens=64,        # name only needs ~30 tokens
    max_pixels=256 * 28 * 28, # ~200K pixels — enough for card text
    min_pixels=4 * 28 * 28,
)

first_time = time.time() - t1
print(f"   First scan time: {first_time:.1f}s")

print("\n   Detected text:")
for res in output1:
    data = res.json
    if callable(data):
        data = data()
    plist = data.get("res", data).get("parsing_res_list", [])
    for block in plist:
        label = block.get("block_label", "")
        content = block.get("block_content", "").strip()
        if content:
            print(f"     [{label}] {content!r}")

print("\n[3/3] Second scan (warm cache)...")
t2 = time.time()

output2 = pipeline.predict(
    IMAGE_PATH,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_chart_recognition=False,
    use_seal_recognition=False,
    max_new_tokens=64,
    max_pixels=256 * 28 * 28,
    min_pixels=4 * 28 * 28,
)

second_time = time.time() - t2
print(f"   Second scan time: {second_time:.1f}s")

print("\n" + "=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)
print(f"  Model load time  : {load_time:.1f}s")
print(f"  First scan       : {first_time:.1f}s")
print(f"  Second scan      : {second_time:.1f}s")
print("=" * 60)
