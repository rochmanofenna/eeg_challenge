# EEG Challenge Docker Container
# For ultimate reproducibility across any platform

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY setup.py .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e .

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p data results checkpoints logs

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python", "train_s3_extended.py", "--help"]

# Usage:
# docker build -t eeg-challenge .
# docker run --gpus all -v $(pwd)/results:/app/results eeg-challenge python train_s3_extended.py --pretrain_epochs 10