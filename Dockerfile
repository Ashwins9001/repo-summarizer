FROM python:3.11-slim

WORKDIR /app

# Use HTTPS for Debian repositories
RUN apt-get update && \
    apt-get install -y --no-install-recommends git build-essential ca-certificates && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app

ENTRYPOINT ["python", "app/main.py"]
