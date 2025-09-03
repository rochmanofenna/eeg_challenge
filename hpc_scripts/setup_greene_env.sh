#!/bin/bash
# Setup script for Greene HPC cluster environment

echo "🔧 Setting up EEG training environment on Greene..."

# Load required modules
module purge
module load cuda/11.8
module load cudnn/8.6.0
module load python/3.10
module load git/2.30.1

# Create virtual environment
python -m venv ~/envs/eeg_env
source ~/envs/eeg_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 11.8 support
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install project dependencies
pip install numpy==1.24.3
pip install mne==1.5.1
pip install braindecode==0.8.1
pip install scikit-learn scipy pandas
pip install autoreject pyriemann
pip install s3fs boto3
pip install tqdm wandb

# Create necessary directories
mkdir -p $HOME/eeg_challenge
mkdir -p $SCRATCH/checkpoints
mkdir -p $SCRATCH/data_cache
mkdir -p $HOME/eeg_challenge/results
mkdir -p $HOME/eeg_challenge/logs

echo "✅ Environment setup complete!"
echo "📍 Project directory: $HOME/eeg_challenge"
echo "💾 Checkpoint directory: $SCRATCH/checkpoints"
echo "📊 Results directory: $HOME/eeg_challenge/results"

# Instructions
echo ""
echo "📋 Next steps:"
echo "1. Copy your code to: $HOME/eeg_challenge"
echo "2. Submit job: sbatch hpc_scripts/submit_greene.sbatch"
echo "3. Monitor: squeue -u $USER"
echo "4. Check logs: tail -f logs/eeg_train_*.out"