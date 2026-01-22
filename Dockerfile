FROM python:3.11-slim

WORKDIR /app

# Install git and build tools
RUN apt-get update && \
    apt-get install -y git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code
COPY app/ app/
COPY data/ data/

ENV PYTHONUNBUFFERED=1

CMD ["python", "app/main.py"]



