PY=python

.PHONY: setup train-v01 run test lint docker-build docker-run

setup:
	pip install -r requirements-dev.txt

train-v01:
	$(PY) -m ml.train --seed 42 --output-dir artifacts --version v0.1
	mkdir -p app/model && cp artifacts/pipeline.pkl app/model/ && cp artifacts/feature_names.json app/model/
	echo v0.1 > app/model/MODEL_VERSION

run:
	uvicorn app.app:app --host 0.0.0.0 --port 8080

test:
	pytest -q

lint:
	ruff check .

docker-build:
	docker build -t ghcr.io/ORG/REPO:local .

docker-run:
	docker run -p 8080:8080 ghcr.io/ORG/REPO:local

train-v02:
	python -m ml.train --model ridge --seed 42 --output-dir artifacts --version v0.2
	mkdir -p app/model && cp artifacts/pipeline.pkl app/model/ && cp artifacts/feature_names.json app/model/
	echo v0.2 > app/model/MODEL_VERSION