FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

ENV DOCS_URL=/docs REDOC_URL=/redoc OPENAPI_URL=/openapi.json
EXPOSE 8080

HEALTHCHECK CMD python -c "import urllib.request, json; print(json.load(urllib.request.urlopen('http://127.0.0.1:8080/health'))['status'])" || exit 1

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8080"]
