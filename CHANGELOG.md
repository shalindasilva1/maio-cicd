# Changelog
## v0.2
- Model: **Ridge(alpha=1.0)** with StandardScaler (seed=42).
- Rationale: L2 regularization to reduce variance vs. plain LinearRegression.
- Results: RMSE **<fill with CI value>** (↓ from v0.1). See `metrics.json` in the GitHub Release assets.
- API: unchanged (`POST /predict` returns `{"prediction": <float>}`).
## v0.1
- Baseline: StandardScaler + LinearRegression (seed=42).
- Results: RMSE reported in Release assets (`metrics.json`).
- Ships working API & Docker image.
