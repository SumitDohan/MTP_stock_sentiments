FROM python:3.12-slim

# Avoid Python writing pyc files & buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies and tini for process management
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    libffi-dev \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir mlflow dvc

# Copy project code
COPY . .

# Optional: Pull DVC data/models if remote is configured
RUN dvc pull --quiet || echo "No DVC remote configured"

# Expose FastAPI and MLflow UI ports
EXPOSE 8000 5000

# Use tini as entrypoint to manage child processes properly
ENTRYPOINT ["/usr/bin/tini", "--"]

# Start both FastAPI and MLflow UI
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port 8000 & mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri /app/mlruns"]
