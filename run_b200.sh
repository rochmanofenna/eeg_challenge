#!/bin/bash
# B200 Ultra-Fast Training - Deploy in 30 seconds

echo "🚀 B200 ULTRA-FAST TRAINING (Target: 30-40 minutes)"
echo "================================================="

# Quick setup (skip if already done)
if [ ! -d "venv" ]; then
    echo "⚡ Quick setup..."
    python3 -m venv venv
    source venv/bin/activate
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
else
    echo "✅ Using existing environment"
    source venv/bin/activate
fi

# Set B200 optimal settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1

# Create directories
mkdir -p checkpoints results logs

echo ""
echo "🔥 Launching 2x B200 training..."
echo "Expected completion: $(date -d '+40 minutes' '+%H:%M')"
echo ""

# Launch ultra-fast training
python train_b200_ultra.py --gpus 2

echo ""
echo "✅ B200 training completed!"
echo "Check results in: checkpoints/b200_final_model.pt"