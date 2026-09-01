# Asthra ID Scanner API

High-performance **Hybrid ID Card Recognition Backend** for the **Asthra College Welcome Robot**.

This service accepts uploaded college ID card images, reads text via an intelligent **hybrid dual-engine OCR pipeline**, extracts the visitor's full legal name, applies honorific and VIP correction logic, and returns personalized welcome greetings in real time.

---

## 📋 Table of Contents

- [Overview & Hybrid Architecture](#-overview--hybrid-architecture)
- [Key Features](#-key-features)
- [System Requirements](#-system-requirements)
- [Step-by-Step Installation (Fresh Windows Laptop)](#-step-by-step-installation-fresh-windows-laptop)
- [Running the Backend](#-running-the-backend)
- [API Endpoints](#-api-endpoints)
- [Testing the Service](#-testing-the-service)
- [Performance Benchmarks](#-performance-benchmarks)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Important Technical Notes](#-important-technical-notes)

---

## ⚡ Overview & Hybrid Architecture

The Asthra ID Scanner uses a **two-tier hybrid architecture** to balance ultra-fast processing with maximum accuracy across clear and difficult ID card images:

```
                  ┌───────────────────────────────┐
                  │    POST /scan-id (ID Image)   │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │     OpenCV Preprocessing      │
                  │ (Resize + CLAHE Enhancement)  │
                  └──────────────┬────────────────┘
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │    Tier 1: EasyOCR Fast Path  │
                  │   (English CPU Neural Engine) │
                  └──────────────┬────────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 │ Confidence >= 0.75 and Name?  │
                 └───────┬───────────────┬───────┘
                         │               │
                  YES    │               │ NO (Uncertain / Low Conf)
                         ▼               ▼
        ┌─────────────────────────┐  ┌─────────────────────────┐
        │  Immediate Fast Return  │  │ Tier 2: PaddleOCR-VL    │
        │   (~2.0 - 2.5 seconds)  │  │ Vision-Language Fallback│
        └─────────────────────────┘  └───────────┬─────────────┘
                                                 │
                                                 ▼
                                     ┌─────────────────────────┐
                                     │  Accurate Final Result  │
                                     └─────────────────────────┘
```

1. **Tier 1 (Fast Path — EasyOCR)**:
   - Evaluates the image using CRAFT text detection and deep CRNN recognition.
   - If a confident person name is detected ($\text{confidence} \ge 0.75$, $\text{blended score} \ge 0.75$), the result is returned immediately in **~2 to 3 seconds**.
2. **Tier 2 (Fallback Path — PaddleOCR-VL 1.6)**:
   - If Tier 1 fails, detects low-confidence text, or encounters severe distortion, the system seamlessly falls back to **PaddleOCR-VL 1.6** (a 0.9B Vision-Language model) to extract the name with high contextual comprehension.

---

## ✨ Key Features

- **Hybrid Dual-Engine OCR**: Sub-3-second fast path with automatic fallback for challenging photos.
- **Intelligent Name Extraction**: Rule-based heuristic scoring engine that filters out college headers, department labels, course codes (e.g., `B.Tech ECS`), registration numbers (e.g., `24ES031`), dates, and signatures.
- **VIP Name Normalization**: Fuzzy string matcher that corrects minor OCR typos for institutional VIPs, dignitaries, and department heads.
- **Lossless Orientation Handling**: Tests $0^\circ, 90^\circ, 180^\circ,$ and $270^\circ$ rotations with quality gating to correct rotated ID cards without unnecessary computation.
- **Image Preprocessing**: CLAHE (Contrast Limited Adaptive Histogram Equalization) and dimension normalization for uneven camera lighting.
- **MD5 Duplicate Frame Caching**: Prevents redundant OCR recomputation if identical frames are uploaded repeatedly.
- **Production-Ready FastAPI**: Fully typed endpoints, lifespan model management, CORS configured for web and robot clients, and interactive Swagger documentation.

---

## 💻 System Requirements

The backend is engineered to run on standard laptop hardware without requiring a dedicated high-end GPU.

| Component | Minimum Requirement | Recommended Specification |
|---|---|---|
| **Operating System** | Windows 10 / 11 (64-bit) | Windows 11 (64-bit) |
| **Python** | Python 3.10.x (64-bit) | Python 3.10.6 – 3.10.11 |
| **CPU** | 4 cores / 4 threads (Intel i5 6th Gen+) | 4+ cores / 8+ threads (Intel i7 / AMD Ryzen 5+) |
| **RAM** | 8 GB | 16 GB – 24 GB *(both models reside in memory)* |
| **Storage** | 6 GB free disk space | SSD with 10 GB free disk space |
| **Network** | Internet access required on first startup | Stable broadband connection |

> **Disk Space Breakdown**:
> - Virtual environment dependencies (`.venv`): **~1.7 GB**
> - Downloaded AI models (`~/.paddlex` & `~/.EasyOCR`): **~2.4 GB**

---

## 🚀 Step-by-Step Installation (Fresh Windows Laptop)

Follow these exact steps in **Windows PowerShell** to configure the project on a new laptop from scratch.

### Step 1: Install Python 3.10

1. Download **Python 3.10.x (64-bit)** from [python.org](https://www.python.org/downloads/release/python-31011/).
2. Run the installer and **ensure you check**:
   - ☑ **"Add Python 3.10 to PATH"**
3. Open a new PowerShell window and verify:
   ```powershell
   python --version
   ```
   *(Should output `Python 3.10.x`)*

---

### Step 2: Clone or Copy the Repository

Navigate to your workspace directory and clone the repository:

```powershell
git clone https://github.com/dennis-sabu/asthra_fast_api.git
cd asthra_fast_api
```

---

### Step 3: Configure PowerShell Execution Policy & Create Virtual Environment

1. Allow local script execution for the virtual environment:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   ```

2. Create a dedicated Python virtual environment:
   ```powershell
   python -m venv .venv
   ```

3. Activate the virtual environment:
   ```powershell
   .\.venv\Scripts\activate
   ```
   *(Your terminal prompt will now show `(.venv)` at the beginning.)*

---

### Step 4: Install Dependencies

Upgrade `pip` and install all required libraries from `requirements.txt`:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Note**: This installs PyTorch (CPU), EasyOCR, PaddlePaddle (CPU), PaddleOCR, PaddleX, OpenCV, Pillow, FastAPI, and Uvicorn in one step.

---

### Step 5: Initial Model Download (Automatic on First Startup)

When the application starts for the first time, it will automatically download the necessary AI models:
- **EasyOCR Models** (`~/.EasyOCR/model/`): CRAFT text detector (~80 MB) and English recognition model (~14 MB).
- **PaddleOCR-VL Models** (`~/.paddlex/official_models/`): PaddleOCR-VL 1.6 VLM weights & layout models (~1.8 GB).

Ensure your laptop is connected to the internet during the very first run.

---

## 🔌 Running the Backend

Ensure your virtual environment is active (`(.venv)` showing in prompt), then start the server:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Direct Invocation (Without Activating Shell):
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Startup Output:
```text
========================================================
 Asthra ID Scanner - starting up
========================================================
 Loading PaddleOCR-VL 1.6...
 PaddleOCR-VL loaded in 19.8s  |  device=cpu
 Loading LightOCRScanner for fast-path...
 EasyOCR loaded in 4.5s

 Startup complete in 24.3s
 VL model loaded:     True
 Device:              cpu
 LightOCR:            ready
========================================================
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

The API is now live at: **`http://127.0.0.1:8000`**

---

## 📡 API Endpoints

### 1. Root Information
- **Method**: `GET`
- **Path**: `/`
- **Purpose**: Verifies that the service is alive.
- **Response**:
  ```json
  {
    "service": "Asthra ID Scanner",
    "status": "running",
    "model": "PaddleOCR-VL-1.6"
  }
  ```

---

### 2. Health & Model Status
- **Method**: `GET`
- **Path**: `/health`
- **Purpose**: Returns health status, loaded AI model verification, and active inference device (`cpu` or `gpu`).
- **Response**:
  ```json
  {
    "status": "healthy",
    "model_loaded": true,
    "device": "cpu",
    "model": "PaddleOCR-VL-1.6"
  }
  ```

---

### 3. Scan ID Card (Main Hybrid Endpoint)
- **Method**: `POST`
- **Path**: `/scan-id`
- **Purpose**: Scans an uploaded ID card image through the Hybrid OCR pipeline and extracts the person's full name.
- **Request Format**: `multipart/form-data`
  - Field: `file`
  - Supported types: `.jpg`, `.jpeg`, `.png`, `.webp` (Max size: 10 MB)

#### Successful Scan Response (`200 OK`):
```json
{
  "success": true,
  "name": "Dennis Sabu",
  "message": "Welcome, Dennis Sabu!",
  "confidence": 0.997,
  "processing_time_ms": 2036
}
```

#### Unreadable ID Response (`200 OK`):
```json
{
  "success": false,
  "name": null,
  "message": "Couldn't read the name clearly. Please try again.",
  "confidence": 0.0,
  "processing_time_ms": 2150
}
```

#### Validation Error (`400 Bad Request`):
```json
{
  "detail": "Unsupported file type. Please upload a JPG, PNG, or WEBP image."
}
```

---

### 4. Test Lightweight OCR (Direct EasyOCR Endpoint)
- **Method**: `POST`
- **Path**: `/test-paddle-ocr`
- **Purpose**: Direct experimental endpoint to inspect EasyOCR raw detection boxes, timing breakdowns, and confidence values without invoking the VLM fallback.
- **Request Format**: `multipart/form-data` (`file`)
- **Response**:
  ```json
  {
    "success": true,
    "name": "Dennis Sabu",
    "message": "Welcome, Dennis Sabu!",
    "raw_text": [
      "ST. JOSEPH'S COLLEGE...",
      "Dennis Sabu",
      "B.Tech ECS 2024-28"
    ],
    "ocr_confidence": 0.997,
    "candidate_score": 0.700,
    "blended_score": 0.849,
    "timing": {
      "preprocessing_ms": 16,
      "ocr_ms": 2018,
      "name_extraction_ms": 0,
      "total_ms": 2034
    }
  }
  ```

---

## 🧪 Testing the Service

### Option 1: Interactive Swagger UI (Browser)
1. Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser.
2. Expand `POST /scan-id`.
3. Click **Try it out**.
4. Select an ID card image (`id.jpeg`) and click **Execute**.

---

### Option 2: Automated API Test Suite
Run the automated end-to-end test suite in a separate terminal:

```powershell
.\.venv\Scripts\activate
python test_api.py
```

This verifies:
- Service discovery (`GET /`)
- Health and device status (`GET /health`)
- Normal ID scanning (`POST /scan-id`)
- Frame caching speedup
- File type validation & error responses
- VIP name correction unit tests

---

### Option 3: Standalone Batch OCR Benchmark
Run the standalone OCR evaluation tool on all test images:

```powershell
python benchmark_ocr.py
```

**Benchmark Options**:
- Test specific file: `python benchmark_ocr.py --images id.jpeg`
- Measure with expected name: `python benchmark_ocr.py --expected "id.jpeg=DENNIS SABU"`
- Compare preprocessing variants: `python benchmark_ocr.py --preproc-compare`
- Test multi-angle orientation: `python benchmark_ocr.py --rotations`

---

### Option 4: PowerShell `Invoke-RestMethod`
```powershell
$form = @{ file = Get-Item "id.jpeg" }
Invoke-RestMethod -Uri "http://127.0.0.1:8000/scan-id" -Method Post -Form $form
```

### Option 5: cURL
```bash
curl -X POST "http://127.0.0.1:8000/scan-id" -F "file=@id.jpeg"
```

---

## 📊 Performance Benchmarks

Measured on Intel Core i7 (4 cores / 8 threads, CPU-only inference):

| Pipeline Stage | Fast Path (EasyOCR) | Fallback Path (PaddleOCR-VL) |
|---|---|---|
| **Image Preprocessing** | 10 – 20 ms | 150 – 250 ms |
| **Neural Inference** | 1,850 – 2,100 ms | 18,000 – 22,000 ms |
| **Name Parsing & Scoring** | < 1 ms | < 5 ms |
| **Total Response Time (Warm)** | **~2.0 – 2.5 seconds** | **~20 – 22 seconds** |
| **Duplicate Frame Cache Hit** | **< 2 ms** | **< 2 ms** |

---

## 📁 Project Structure

```text
asthra_fast_api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI server, lifespan loader, CORS, route handlers
│   ├── scanner.py           # PaddleOCR-VL 1.6 VLM fallback scanner & VIP dictionary
│   ├── scanner_ocr.py       # LightOCRScanner (EasyOCR fast path engine & scoring)
│   └── preprocessing.py     # OpenCV pipeline: CLAHE, resize, unsharp mask, lossless rotation
├── test_images/             # Folder containing sample ID card images for testing
├── benchmark_ocr.py         # Standalone batch benchmark & preprocessing comparison tool
├── benchmark_vl.py          # Standalone PaddleOCR-VL speed benchmark
├── test_api.py              # End-to-end integration & unit test suite
├── id.jpeg                  # Sample ID card image
├── requirements.txt         # Consolidated Python dependencies
└── README.md                # Project documentation
```

---

## 🛠️ Troubleshooting

### 1. `python` or `pip` is not recognized
- **Cause**: Python was installed without adding to system `PATH`.
- **Solution**: Re-run the Python installer, choose **Modify**, and check **Add Python to environment variables**. Or invoke via full path: `C:\Users\<User>\AppData\Local\Programs\Python\Python310\python.exe`.

### 2. PowerShell Script Execution Disabled
- **Symptom**: `cannot be loaded because running scripts is disabled on this system`.
- **Solution**: Execute the following command in PowerShell:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
  ```
  Then re-activate: `.\.venv\Scripts\activate`.

### 3. Port 8000 is Already in Use
- **Symptom**: `[Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000)`.
- **Solution A**: Identify and stop the process using port 8000:
  ```powershell
  Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force
  ```
- **Solution B**: Run on an alternate port:
  ```powershell
  python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
  ```

### 4. Model Download Failed / Connection Timed Out
- **Symptom**: `requests.exceptions.ConnectionError` or download stuck on first launch.
- **Solution**: Verify internet access. If behind a proxy or college firewall, set proxy environment variables or hotspot your laptop for the initial download. Once downloaded to `~/.paddlex` and `~/.EasyOCR`, the system works completely **offline**.

### 5. Frontend Cannot Connect / CORS Errors
- **Symptom**: Browser reports `Cross-Origin Request Blocked`.
- **Solution**: Ensure your frontend URL is registered in `ALLOWED_ORIGINS` inside [`app/main.py`](file:///c:/Users/denni/OneDrive/Documents/vs%20code/websites/asthra_fast_api/app/main.py). Current defaults allow `http://localhost:3000`, `http://127.0.0.1:3000`, and `https://asthra-welcome-robot.vercel.app`.

### 6. High RAM Usage / Out of Memory
- **Symptom**: System slowdown on 8 GB RAM laptops.
- **Solution**: Close heavy background applications (Chrome tabs, IDEs). The system uses ~2.5 GB RAM total with both models loaded in memory.

---

## 📌 Important Technical Notes

1. **Model Persistence**: Models are loaded **once** at server startup and retained in memory. Individual scan requests do not incur model initialization overhead.
2. **First Request vs Warm Request**: The very first request on a newly booted server may take slightly longer due to JIT compilation. Warm scans stabilize at ~2 seconds.
3. **No GPU Required**: The default installation utilizes multi-threaded CPU matrix operations (oneDNN & PyTorch CPU kernels).
4. **Safety Guarantee**: The existing production scanner ([`app/scanner.py`](file:///c:/Users/denni/OneDrive/Documents/vs%20code/websites/asthra_fast_api/app/scanner.py)) is preserved and acts as an automatic safety net whenever the fast engine encounters ambiguous data.
