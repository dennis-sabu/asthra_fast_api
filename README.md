# Asthra ID Scanner API

FastAPI backend service for the **Asthra College Welcome Robot**. This service accepts uploaded ID card images, performs Optical Character Recognition (OCR) using **PaddleOCR**, extracts visitor names, applies VIP correction logic, and returns greeting payloads.

---

## 📋 Table of Contents

- [Prerequisites](#-prerequisites)
- [Step-by-Step Local Setup & Execution](#-step-by-step-local-setup--execution)
  - [1. Open Terminal in Project Directory](#1-open-terminal-in-project-directory)
  - [2. Create & Activate Virtual Environment](#2-create--activate-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Start the FastAPI Server](#4-start-the-fastapi-server)
- [API Endpoints & Testing](#-api-endpoints--testing)
  - [Interactive API Documentation](#interactive-api-documentation)
  - [Running the Test Suite](#running-the-test-suite)
- [Troubleshooting Common Issues](#-troubleshooting-common-issues)

---

## ⚙️ Prerequisites

- **Python**: Version 3.10 or higher installed on your system.
- **Operating System**: Windows / Linux / macOS.

---

## 🚀 Step-by-Step Local Setup & Execution

### 1. Open Terminal in Project Directory

Open PowerShell, Command Prompt, or VS Code integrated terminal in the project root folder:



---

### 2. Create & Activate Virtual Environment

#### Create environment (if not created yet):
```powershell
python -m venv .venv
```

#### Activate the environment:

- **Windows (PowerShell)**:
  ```powershell
  .\.venv\Scripts\activate
  ```
- **Windows (Command Prompt / CMD)**:
  ```cmd
  .\.venv\Scripts\activate.bat
  ```
- **Linux / macOS**:
  ```bash
  source .venv/bin/activate
  ```

*(Once activated, your terminal prompt should display `(.venv)` at the beginning of the line.)*

---

### 3. Install Dependencies

Install all required packages into the activated environment:

```powershell
pip install -r requirements.txt
```

*Note: For GPU acceleration with CUDA, refer to [PaddlePaddle Installation Guide](https://www.paddlepaddle.org.cn/install/quick).*

---

### 4. Start the FastAPI Server

#### Option A: Recommended (With Environment Activated)

Run Uvicorn server using the activated Python environment:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

#### Option B: Direct Execution (Without activating venv prompt)

You can invoke the virtual environment's Python executable directly:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 🔗 API Endpoints & Testing

Once the server is running on `http://127.0.0.1:8000`:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Root service info |
| `GET` | `/health` | Health check & model status |
| `POST` | `/scan-id` | Upload ID image (multipart/form-data) |

### Interactive API Documentation

- **Swagger UI**: Visit [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser to test endpoints interactively.
- **ReDoc**: Visit [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc).

### Running the Test Suite

Open a **new terminal window**, activate the virtual environment, and run:

```powershell
.\.venv\Scripts\activate
python test_api.py
```

---

## 🛠️ Troubleshooting Common Issues

### Issue 1: `python : The term 'python' is not recognized...`

- **Cause**: Python is not added to your system `PATH`, or the virtual environment is not activated in your current terminal session.
- **Solution**:
  1. Activate the environment first: `.\.venv\Scripts\activate`
  2. Or specify the direct venv path: `.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000`

### Issue 2: PowerShell script execution error (`cannot be loaded because running scripts is disabled`)

- **Cause**: PowerShell restriction on running local scripts.
- **Solution**: Run this command once in PowerShell to allow running local environment scripts:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
  ```
  Then re-run `.\.venv\Scripts\activate`.
