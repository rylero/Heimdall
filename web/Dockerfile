FROM node:20-slim AS frontend-build
WORKDIR /build
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y v4l-utils && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir .

COPY app/ ./app/
COPY --from=frontend-build /build/dist ./frontend/dist

ENV HEIMDALL_CONFIG_DIR=/app/config
ENV HEIMDALL_LOGS_DIR=/app/logs
ENV HEIMDALL_CONTAINER=heimdall

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
