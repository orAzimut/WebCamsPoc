FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Extra index so pip can fetch the cu121 wheels
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip3 install --no-cache-dir --extra-index-url ${TORCH_INDEX} -r requirements.txt

# Project files & YOLO weights
COPY . .
RUN mkdir -p /app/models
COPY yolo11n.pt /app/models/yolo11n.pt

ENV PYTHONPATH=/app
CMD ["python3", "src/main.py"]
