FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install deps in stages — heavier packages first for better layer caching
COPY requirements-docker.txt .
RUN pip install --no-cache-dir numpy pandas scipy && \
    pip install --no-cache-dir -r requirements-docker.txt

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "backend.main"]
