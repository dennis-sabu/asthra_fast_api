"""
test_api.py — Test suite for the Asthra ID Scanner API

Tests:
    1. GET /          — service info
    2. GET /health    — model status and device
    3. POST /scan-id  — scan id.jpeg, confirm "Dennis Sabu"
    4. POST /scan-id  — repeated scan (should hit cache, be faster)
    5. POST /scan-id  — invalid file type (expect 400)
    6. POST /scan-id  — missing file (expect 422)
    7. VIP correction — unit tests on name correction logic

Usage:
    # Start the server first:
    .venv\\Scripts\\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000

    # Then in another terminal:
    .venv\\Scripts\\python.exe test_api.py
"""

import sys
import time

import requests

BASE_URL = "http://127.0.0.1:8000"
ID_IMAGE_PATH = "id.jpeg"

PASS = "PASS"
FAIL = "FAIL"
passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        print(f"  {PASS}  {label}")
        passed += 1
    else:
        print(f"  {FAIL}  {label}  <- {detail}")
        failed += 1


def section(title: str):
    print()
    print("-" * 54)
    print(f"  {title}")
    print("-" * 54)


# ----------------------------------------------------------
# 1. GET /
# ----------------------------------------------------------
section("1. GET /")
try:
    r = requests.get(f"{BASE_URL}/", timeout=5)
    body = r.json()
    check("Status 200", r.status_code == 200, f"got {r.status_code}")
    check("service = 'Asthra ID Scanner'", body.get("service") == "Asthra ID Scanner", str(body))
    check("status = 'running'", body.get("status") == "running", str(body))
    check("model field present", "model" in body, str(body))
except Exception as e:
    check("GET / reachable", False, str(e))

# ──────────────────────────────────────────────────────────
# 2. GET /health
# ──────────────────────────────────────────────────────────
section("2. GET /health")
try:
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    body = r.json()
    check("Status 200", r.status_code == 200, f"got {r.status_code}")
    check("status = 'healthy'", body.get("status") == "healthy", str(body))
    check("model_loaded = true", body.get("model_loaded") is True, str(body))
    check("device field present", "device" in body, str(body))
    device = body.get("device", "unknown")
    check(f"device = '{device}'", device in ("cpu", "gpu"), str(body))
    print(f"      → device reported: {device}")
except Exception as e:
    check("GET /health reachable", False, str(e))

# ──────────────────────────────────────────────────────────
# 3. POST /scan-id  (first scan — cold)
# ──────────────────────────────────────────────────────────
section("3. POST /scan-id  (first scan)")
try:
    with open(ID_IMAGE_PATH, "rb") as f:
        t0 = time.time()
        r = requests.post(
            f"{BASE_URL}/scan-id",
            files={"file": ("id.jpeg", f, "image/jpeg")},
            timeout=300,
        )
        elapsed_ms = int((time.time() - t0) * 1000)

    body = r.json()
    check("Status 200", r.status_code == 200, f"got {r.status_code}")
    check("success = true", body.get("success") is True, str(body))
    check("name = 'Dennis Sabu'", body.get("name") == "Dennis Sabu", f"got {body.get('name')!r}")
    check("message contains name", "Dennis Sabu" in body.get("message", ""), str(body))
    check("confidence > 0", body.get("confidence", 0) > 0, str(body))
    check("processing_time_ms present", "processing_time_ms" in body, str(body))
    print(f"      → name:        {body.get('name')!r}")
    print(f"      → confidence:  {body.get('confidence')}")
    print(f"      → time (API):  {elapsed_ms}ms")
    print(f"      → time (svc):  {body.get('processing_time_ms')}ms")
except Exception as e:
    check("scan-id first scan", False, str(e))

# ──────────────────────────────────────────────────────────
# 4. POST /scan-id  (repeated — cache hit)
# ──────────────────────────────────────────────────────────
section("4. POST /scan-id  (repeated scan — cache)")
try:
    with open(ID_IMAGE_PATH, "rb") as f:
        t0 = time.time()
        r = requests.post(
            f"{BASE_URL}/scan-id",
            files={"file": ("id.jpeg", f, "image/jpeg")},
            timeout=300,
        )
        elapsed_ms = int((time.time() - t0) * 1000)

    body = r.json()
    check("Status 200", r.status_code == 200, f"got {r.status_code}")
    check("success = true", body.get("success") is True, str(body))
    check("name = 'Dennis Sabu'", body.get("name") == "Dennis Sabu", f"got {body.get('name')!r}")
    check("fast response (< 5s)", elapsed_ms < 5000, f"took {elapsed_ms}ms")
    print(f"      → time (API):  {elapsed_ms}ms  ← should be near-instant")
except Exception as e:
    check("scan-id repeated scan", False, str(e))

# ──────────────────────────────────────────────────────────
# 5. Invalid file type
# ──────────────────────────────────────────────────────────
section("5. POST /scan-id  (invalid file type)")
try:
    r = requests.post(
        f"{BASE_URL}/scan-id",
        files={"file": ("test.txt", b"not an image", "text/plain")},
        timeout=10,
    )
    check("Status 400", r.status_code == 400, f"got {r.status_code}")
    check("No traceback in response", "Traceback" not in r.text, r.text[:200])
except Exception as e:
    check("invalid type rejected", False, str(e))

# ──────────────────────────────────────────────────────────
# 6. Missing file field
# ──────────────────────────────────────────────────────────
section("6. POST /scan-id  (missing file)")
try:
    r = requests.post(f"{BASE_URL}/scan-id", timeout=5)
    check("Status 422", r.status_code == 422, f"got {r.status_code}")
except Exception as e:
    check("missing file rejected", False, str(e))

# ──────────────────────────────────────────────────────────
# 7. VIP correction unit tests (no server needed)
# ──────────────────────────────────────────────────────────
section("7. VIP correction logic  (unit tests)")
sys.path.insert(0, ".")
try:
    from app.scanner import apply_vip_correction, is_valid_name_candidate

    cases = [
        # (ocr_input,                      expected_output,                      should_correct)
        ("Dr. V. P. Devassla",             "Dr. V. P. Devassia",                 True),
        ("Dr V P Devassia",                "Dr. V. P. Devassia",                 True),
        ("Dr. Giby Jose",                  "Dr. Giby Jose",                      True),
        ("Dr Giby Jose",                   "Dr. Giby Jose",                      True),
        ("Rev Prof Dr James John Mangalathu", "Rev. Prof. Dr. James John Mangalathu", True),
        # Non-VIP names must NOT be forced to a VIP
        ("John Mathew",                    "John Mathew",                        False),
        ("Anu Joseph",                     "Anu Joseph",                         False),
        ("Rahul Thomas",                   "Rahul Thomas",                       False),
    ]

    for ocr_in, expected, should_correct in cases:
        corrected, was_corrected = apply_vip_correction(ocr_in)
        if should_correct:
            check(
                f"VIP correction: {ocr_in!r} → {expected!r}",
                corrected == expected,
                f"got {corrected!r}",
            )
        else:
            check(
                f"No forced VIP: {ocr_in!r} stays unchanged",
                corrected == expected and not was_corrected,
                f"got {corrected!r} (corrected={was_corrected})",
            )

    # Validator tests
    valid_names = [
        "Dennis Sabu",
        "Dr. V. P. Devassia",
        "Rev. Prof. Dr. James John Mangalathu",
        "Msgr. Dr. Joseph Thadathil",
        "John Mathew",
        "Anu Joseph",
    ]
    invalid_names = [
        "24ES031",          # registration number
        "B.Tech ECS",       # course code
        "AUTONOMOUS",       # institution label
        "2024-28",          # date range
        "ST.JOSEPH'S",      # college name
    ]
    for name in valid_names:
        check(f"Valid: {name!r}", is_valid_name_candidate(name), "rejected unexpectedly")
    for name in invalid_names:
        check(f"Reject: {name!r}", not is_valid_name_candidate(name), "accepted unexpectedly")

except Exception as e:
    check("VIP unit tests", False, str(e))
    import traceback
    traceback.print_exc()

# ──────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────
print()
print("=" * 54)
total = passed + failed
print(f"  RESULTS: {passed}/{total} passed  |  {failed} failed")
print("=" * 54)
print()

if failed > 0:
    sys.exit(1)
