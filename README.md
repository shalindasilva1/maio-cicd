# Virtual Diabetes Clinic – ML API 🩺

A lightweight FastAPI service that serves a regression model trained on the diabetes dataset.  
Every tagged release automatically builds and publishes a Docker image to **GitHub Container Registry (GHCR)**.

---

## 🐳 Run with Docker (recommended)

Published image lives at:

```
ghcr.io/shalindasilva1/maio-cicd
```

### Pull and run the container

```bash
docker pull ghcr.io/shalindasilva1/maio-cicd:latest
docker run --rm -p 8080:8080 ghcr.io/shalindasilva1/maio-cicd:latest
```

The API will be available at **http://localhost:8080**

### 3. Test the API

```bash
# Health check
curl -s http://localhost:8080/health

# Prediction example
curl -s -X POST http://localhost:8080/predict   -H "Content-Type: application/json"   -d '{"age":0.03,"sex":-0.04,"bmi":0.02,"bp":0.01,"s1":-0.02,"s2":0.01,"s3":0.0,"s4":-0.01,"s5":0.02,"s6":-0.001}'
```

Expected response:
```json
{"prediction": 123.45}
```

You can also explore the interactive docs at:  
👉 [http://localhost:8080/docs](http://localhost:8080/docs)

---

## 🧩 API Endpoints

| Method | Path | Description |
|--------|------|--------------|
| `GET` | `/health` | Returns service status and model version |
| `POST` | `/predict` | Predicts disease progression |
| `GET` | `/docs` | Swagger UI for testing |

---

## 🧪 Run tests (locally)

```bash
pytest -q
```

If you see `ModuleNotFoundError: No module named 'app'`, run:
```bash
PYTHONPATH=. pytest -q
```

---

## ⚙️ Run locally (for development)

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model and prepare artifacts

```bash
python -m ml.train --seed 42 --output-dir artifacts --version v0.1.0
mkdir -p app/model
cp artifacts/pipeline.pkl app/model/
cp artifacts/feature_names.json app/model/
echo v0.1.0 > app/model/MODEL_VERSION
```

### 4. Run the API

```bash
uvicorn app.app:app --host 0.0.0.0 --port 8080
```

Open [http://localhost:8080/docs](http://localhost:8080/docs) to test endpoints.

---

## 🚀 Publishing a new image

Every time you push a **git tag** (like `v0.1.0`), GitHub Actions will:

1. Train and test the model.
2. Build a Docker image.
3. Push it to GHCR as:

```
ghcr.io/shalindasilva1/maio-cicd:v0.1.0
ghcr.io/shalindasilva1/maio-cicd:latest
```

### Example

```bash
git tag v0.1.1
git push origin v0.1.1
```

Once the workflow completes, the image will appear under:  
**https://github.com/shalindasilva1/maio-cicd/pkgs/container/maio-cicd**

---

## 🧠 Notes

- The container exposes port **8080**.
- Model artifacts are baked into the image from `app/model/`.
- CI/CD and release pipelines are defined in `.github/workflows/`.

---
