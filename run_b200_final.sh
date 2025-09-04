#!/bin/bash
# B200 FULL DATASET TRAINING - FINAL VERSION
# For 2x B200 GPUs on RunPod

echo "🚀 B200 FULL DATASET TRAINING"
echo "================================"
echo "Dataset: 3,326 HBN EEG files (S3 streaming)"
echo "Target: 30-45 minutes on 2x B200 GPUs"
echo "================================"

# Environment check
echo "📍 Checking environment..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Setup Python environment
if [ ! -d "venv" ]; then
    echo "📦 Setting up Python environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies if needed
if ! python -c "import torch" 2>/dev/null; then
    echo "📚 Installing dependencies..."
    pip install --upgrade pip
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
fi

# Create necessary directories
mkdir -p checkpoints results logs

# Run B200-optimized training
echo ""
echo "🔥 Starting B200 training..."
echo "Start time: $(date)"
echo "Expected completion: $(date -d '+45 minutes')"
echo ""

# Launch with optimal B200 settings
CUDA_VISIBLE_DEVICES=0,1 python train_b200_full.py \
    --batch_size 2048 \
    --pretrain_epochs 80 \
    --finetune_epochs 30 \
    --lr 0.008 \
    --num_workers 16

echo ""
echo "✅ Training complete!"
echo "End time: $(date)"
echo "Check results: checkpoints/b200_full_dataset_final.pt"