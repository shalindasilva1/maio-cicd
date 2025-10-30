# Virtual Diabetes Clinic – ML API (CI → GHCR)

This repository trains a tiny regression model on the scikit‑learn **diabetes** dataset, bakes the trained artifacts into a FastAPI app, and ships a Docker image to **GitHub Container Registry (GHCR)**.

**You don’t need to clone or build local images to try it.** Pull the container from GHCR and call the API.

---

## TL;DR — Pull, run, and ping the API (no clone required)

> Replace `OWNER/REPO` with your GitHub path (e.g., `octocat/maio-cicd`).  
> If the image/package is **private**, authenticate first (see next section).

```bash
# (If private) authenticate to GHCR with a token that has read:packages
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_GH_USERNAME --password-stdin

# Pull an image tag (examples shown below under “Image tags”)
docker pull ghcr.io/OWNER/REPO:latest

# Run it (serves on port 8080 in the container)
docker run --rm -p 8080:8080 ghcr.io/OWNER/REPO:latest

# In another shell, call the API
curl -s http://localhost:8080/health
curl -s http://localhost:8080/openapi.json | head -n 20

# Predict with raw features (the model expects 10 diabetes features)
curl -s -X POST http://localhost:8080/predict   -H "Content-Type: application/json"   -d '{
        "age": 0.03, "sex": -0.04, "bmi": 0.02, "bp": 0.01,
        "s1": -0.02, "s2": 0.01, "s3": 0.0, "s4": -0.01, "s5": 0.02, "s6": -0.001
      }'
```

Expected response shape:
```json
{"prediction": 123.456}
```

**Endpoints** (FastAPI defaults also available):
- `GET /` → redirects to `/docs`
- `GET /health` → `{"status":"ok","model_version":"<string>"}`
- `POST /predict` → `{"prediction": <float>}`

> Feature names match the sklearn diabetes dataset: `age, sex, bmi, bp, s1, s2, s3, s4, s5, s6`.

---

## Image tags (what you can pull)

This repo uses two workflows:

1. **CI** (`.github/workflows/ci.yml`): runs on pushes/PRs to `main`. It **trains** and **bakes** artifacts into `app/model` for tests, and uploads artifacts, but it does **not** push a Docker image.
2. **Release** (`.github/workflows/release.yml`): runs on git **tags** matching `v*`. It trains, bakes artifacts, builds a Docker image, and **pushes to GHCR**.

The release workflow publishes the image at:
```
ghcr.io/OWNER/REPO:<TAG>
```
where `<TAG>` is your git tag (e.g., `v0.1.0`). You can also create a “moving” tag such as `latest` if you add it to the release workflow (see notes below).

**Examples:**
```bash
# exact release version
docker pull ghcr.io/OWNER/REPO:v0.1.0

# if you add a 'latest' tag in the workflow, you could also:
docker pull ghcr.io/OWNER/REPO:latest
```

> **Make it public (optional):** To allow unauthenticated pulls, go to your repo → **Packages** → your container → **Package settings** → set **Visibility** to **Public**.

---

## How the pipeline works

### Training
`ml/train.py` trains a StandardScaler+LinearRegression pipeline on the sklearn diabetes dataset and writes:
- `artifacts/pipeline.pkl`
- `artifacts/feature_names.json`
- `artifacts/metrics.json`
- optionally `artifacts/MODEL_VERSION` if `--version` is set

The CI and Release workflows then **bake** these files into the image by copying them to `app/model/` before building the Docker image.

### API
`app/app.py` exposes:
- `GET /health` for liveness
- `POST /predict` to score one patient vector (10 features)
- `/docs` & `/openapi.json` are enabled via FastAPI

The app loads `app/model/pipeline.pkl`, `feature_names.json`, and `MODEL_VERSION` at startup.

### Docker
`Dockerfile` builds a minimal image running:
```
uvicorn app.app:app --host 0.0.0.0 --port 8080
```

### CI (`.github/workflows/ci.yml`)
- triggers on pushes and PRs to `main`, and on tags `v*`
- installs deps
- runs unit tests:
  - training smoke test (`tests/test_train_smoke.py`)
  - API contract tests (`tests/test_api_contract.py`)
- trains & stages artifacts into `app/model/` for tests
- uploads `artifacts/` as a build artifact

### Release (`.github/workflows/release.yml`)
- triggers on pushes of tags like `v0.1.0`
- trains & stages artifacts
- builds Docker image
- pushes to GHCR as `ghcr.io/${{ github.repository }}:${{ github.ref_name }}`
- creates a GitHub Release and attaches `artifacts/metrics.json`

> **Want a `latest` tag too?** Add a second `docker/build-push-action` step or extend tags to include `latest`. For example:
> ```yaml
>   - name: Build & push
>     uses: docker/build-push-action@v6
>     with:
>       context: .
>       push: true
>       tags: |
>         ghcr.io/${{ github.repository }}:${{ github.ref_name }}
>         ghcr.io/${{ github.repository }}:latest
> ```

---

## Local development (optional)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt

# Train and bake artifacts
python -m ml.train --seed 42 --output-dir artifacts --version v0.1.0
mkdir -p app/model
cp artifacts/pipeline.pkl app/model/
cp artifacts/feature_names.json app/model/
echo v0.1.0 > app/model/MODEL_VERSION

# Run the API
uvicorn app.app:app --host 0.0.0.0 --port 8080
# open http://localhost:8080/docs
```

Run tests:
```bash
pytest -q
```

---

## Cutting a release (to publish an image)

```bash
# Commit your changes on main and ensure CI is green.
git tag v0.1.0
git push origin v0.1.0
```

When the **Release** workflow completes, you’ll see the image under the repo’s **Packages**. Pull it using:
```
docker pull ghcr.io/OWNER/REPO:v0.1.0
```

---

## Troubleshooting

- **401/403 pulling from GHCR**: If the package is private, log in with a token that has `read:packages`. If it’s public and you still get 401, your Docker may be using an old cached credential; run `docker logout ghcr.io` and try again.
- **Tests failing due to missing artifacts**: Ensure the workflow step that copies `artifacts/` to `app/model/` runs **before** API tests. The provided CI already does this.
- **Port conflicts**: The container exposes **8080**. Change host mapping with `-p HOSTPORT:8080` if needed.
