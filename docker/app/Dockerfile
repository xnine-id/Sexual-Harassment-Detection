# Stage 1: Builder
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04 AS builder

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: Runtime
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libxext6 libgl1 libglib2.0-0 libxrender1 libsm6 \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE ${PORT}
CMD ["python3", "cmd/api.py"]