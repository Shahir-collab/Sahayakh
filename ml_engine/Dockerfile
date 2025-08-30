FROM python:3.10-slim

# Install system dependencies for ML
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy ML engine code
COPY . /app/

# Create model directory
RUN mkdir -p /app/models

EXPOSE 8001

CMD ["python", "ml_server.py"]
