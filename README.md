## GEN AI Project

This repository contains a small **GEN AI Project** with a Python backend and a static HTML frontend.

### Project Structure

- **backend** – FastAPI (or similar) Python backend (`main.py`, `requirements.txt`).
- **frontend** – Static web UI (`index.html`).
- **Assets** – Images used by the frontend.
- **venv** – Local Python virtual environment (ignored by Git via `.gitignore`).

### Prerequisites

- Python 3.10+ installed
- (Optional) A modern browser for the frontend (Chrome, Edge, etc.)
- Git installed

### Backend Setup & Run

From the project root:

```bash
cd backend
python -m venv venv  # optional if you want a fresh venv
venv\Scripts\activate  # on Windows PowerShell
pip install -r requirements.txt
uvicorn main:app --reload
```

The backend will usually be available at `http://127.0.0.1:8000` (or whatever host/port you configured in `main.py`).

### Frontend

Open `frontend/index.html` directly in your browser, or serve it with a simple static server (for example, `python -m http.server` inside the `frontend` folder) if your browser blocks local file requests.

### How to Push This Project to GitHub

1. **Create a new empty repository on GitHub** (do *not* add a README or `.gitignore` there).
2. In PowerShell, from the project root (`C:\Users\Madan\Desktop\GEN AI Project`), run:

```powershell
cd "C:\Users\Madan\Desktop\GEN AI Project"
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```

3. Replace `<your-username>` and `<your-repo-name>` with your actual GitHub username and repository name.


