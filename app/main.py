from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .scanner import IDScanner


# ----------------------------------------
# Paths
# ----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(
    exist_ok=True
)


# ----------------------------------------
# FastAPI
# ----------------------------------------

app = FastAPI(
    title="Asthra ID Scanner API",
    description="ID card scanning service using PaddleOCR-VL 1.6",
    version="1.0.0"
)


# ----------------------------------------
# CORS
# ----------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------
# Load model ONCE
# ----------------------------------------

print()
print("Starting Asthra ID Scanner...")
print()

scanner = IDScanner()

print()
print("Asthra ID Scanner is ready.")
print()


# ----------------------------------------
# Health check
# ----------------------------------------

@app.get("/")
def root():
    return {
        "service": "Asthra ID Scanner",
        "status": "running",
        "model": "PaddleOCR-VL-1.6"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# ----------------------------------------
# Scan ID
# ----------------------------------------

@app.post("/scan-id")
async def scan_id(
    file: UploadFile = File(...)
):

    # Validate file type.
    allowed_types = [
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp"
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, PNG or WEBP image."
        )

    # Unique filename.
    extension = Path(
        file.filename or ""
    ).suffix.lower()

    if not extension:
        extension = ".jpg"

    filename = f"{uuid.uuid4().hex}{extension}"

    image_path = UPLOAD_DIR / filename

    try:

        # Save uploaded image.
        with image_path.open("wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

        print()
        print("========================================")
        print("New ID scan")
        print("File:", filename)
        print("========================================")

        # Run PaddleOCR-VL.
        raw_results = scanner.scan(
            str(image_path)
        )

        # Extract OCR blocks.
        text_blocks = scanner.extract_text_blocks(
            raw_results
        )

        # Extract name.
        name = scanner.extract_name(
            text_blocks
        )

        print("Detected text blocks:")

        for block in text_blocks:
            print(
                " -",
                block["text"]
            )

        print("Detected name:", name)

        if not name:
            return {
                "success": False,
                "name": None,
                "message": "Could not confidently identify a name.",
                "text_blocks": text_blocks
            }

        return {
            "success": True,
            "name": name,
            "message": f"Welcome, {name}!",
            "text_blocks": text_blocks
        }

    except Exception as e:

        print("ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=f"ID scanning failed: {str(e)}"
        )

    finally:

        # Delete uploaded image after processing.
        if image_path.exists():
            image_path.unlink()