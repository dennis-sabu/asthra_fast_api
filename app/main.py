"""
main.py — Asthra ID Scanner FastAPI application

Endpoints:
    GET  /          — service info
    GET  /health    — health + model status + device
    POST /scan-id   — scan an ID card image, return person's name

The IDScanner is loaded ONCE at module import time and reused for
every request.  Do NOT create a new scanner instance inside a route.
"""

import io
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .scanner import IDScanner, MAX_UPLOAD_BYTES

# ---------------------------------------------------------------------------
# CORS — centralised configuration
# ---------------------------------------------------------------------------

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Asthra ID Scanner API",
    description=(
        "ID card scanning service for the Asthra college welcome robot. "
        "POST an ID card image to /scan-id to receive the visitor's name "
        "and a welcome message."
    ),
    version="2.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Load scanner ONCE at startup
# ---------------------------------------------------------------------------

_startup_start = time.time()

print()
print("=" * 56)
print(" Asthra ID Scanner - starting up")
print("=" * 56)

scanner = IDScanner()

_startup_elapsed = time.time() - _startup_start

print()
print(f" Startup complete in {_startup_elapsed:.1f}s")
print(f" Model loaded: {scanner.model_loaded}")
print(f" Device:       {scanner.device}")
print("=" * 56)
print()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/")
def root():
    """Service identification."""
    return {
        "service": "Asthra ID Scanner",
        "status": "running",
        "model": "PaddleOCR-VL-1.6",
    }


@app.get("/health")
def health():
    """Health check — model status and active device."""
    return {
        "status": "healthy",
        "model_loaded": scanner.model_loaded,
        "device": scanner.device,
        "model": "PaddleOCR-VL-1.6",
    }


@app.post("/scan-id")
async def scan_id(file: UploadFile = File(...)):
    """
    Scan an ID card image and return the detected person's name.

    Accepts: multipart/form-data  (field: "file")
    Supported formats: JPEG, PNG, WEBP

    Returns:
        {
            "success": true,
            "name": "Dennis Sabu",
            "message": "Welcome, Dennis Sabu!",
            "confidence": 0.94,
            "processing_time_ms": 1850
        }
    """
    print()
    print("=" * 56)
    print(" ID SCAN REQUEST")
    print("=" * 56)

    # ---- Validate content type ----
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
    content_type = (file.content_type or "").lower().split(";")[0].strip()

    # Allow by extension as fallback (some clients omit content-type)
    if content_type not in allowed_types:
        ext = Path(file.filename or "").suffix.lower()
        ext_map = {".jpg": True, ".jpeg": True, ".png": True, ".webp": True}
        if ext not in ext_map:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Please upload a JPG, PNG, or WEBP image.",
            )

    # ---- Read image bytes ----
    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read uploaded file.")

    # ---- Validate size ----
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024*1024)} MB.",
        )

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ---- Validate it's a real image ----
    try:
        img_check = Image.open(io.BytesIO(image_bytes))
        img_check.verify()
    except Exception:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")

    print(f" File:    {file.filename or 'unnamed'}")
    print(f" Size:    {len(image_bytes) / 1024:.1f} KB")

    # ---- Run scanner ----
    try:
        result = scanner.scan_image_bytes(image_bytes)
    except Exception as exc:
        # Never expose Python tracebacks to the client
        print(f"[ERROR] Unexpected: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Internal error during ID scanning. Please try again.",
        )

    print(f" Result:  success={result['success']}  name={result.get('name')!r}")
    print(f" Time:    {result['processing_time_ms']}ms")
    print("=" * 56)
    print()

    return result