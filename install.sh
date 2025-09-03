#!/bin/bash
# One-command setup for EEG Challenge training anywhere
# Usage: bash install.sh [cuda_version]

set -e

echo "🧠 Setting up EEG Challenge Training Pipeline"
echo "=============================================="

# Detect platform
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macos"
else
    echo "⚠️ Unsupported platform: $OSTYPE"
    exit 1
fi

# CUDA version (default 11.8)
CUDA_VERSION=${1:-"cu118"}
echo "🔧 Using CUDA version: $CUDA_VERSION"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA
echo "🚀 Installing PyTorch with CUDA support..."
if [[ "$CUDA_VERSION" == "cu118" ]]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
elif [[ "$CUDA_VERSION" == "cu121" ]]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
else
    echo "⚠️ Using CPU-only PyTorch (no CUDA)"
    pip install torch torchvision
fi

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# Test installation
echo "🧪 Testing installation..."
python -c "
import torch
import mne
import braindecode
import s3fs
print('✅ PyTorch version:', torch.__version__)
print('✅ CUDA available:', torch.cuda.is_available())
print('✅ MNE version:', mne.__version__)
print('✅ S3FS working:', s3fs.__version__)
"

# Create directories
echo "📁 Creating directories..."
mkdir -p data
mkdir -p results
mkdir -p checkpoints
mkdir -p logs

echo ""
echo "🎉 Setup complete! Next steps:"
echo "================================"
echo ""
echo "1. Activate environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Quick test run:"
echo "   python train_s3_extended.py --pretrain_epochs 1 --finetune_epochs 1"
echo ""
echo "3. Full training:"
echo "   python train_s3_extended.py --pretrain_epochs 100 --finetune_epochs 50"
echo ""
echo "4. Or use the CLI:"
echo "   eeg-train --help"
echo ""
echo "🔥 Ready to train on any machine with GPU!"