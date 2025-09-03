#!/bin/bash
#SBATCH --job-name=eeg_s3_train
#SBATCH --output=logs/eeg_train_%j.out
#SBATCH --error=logs/eeg_train_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@university.edu

# Greene cluster specific modules
module purge
module load cuda/11.8
module load cudnn/8.6.0
module load python/3.10
module load openmpi/4.1.4

# Activate virtual environment
source ~/envs/eeg_env/bin/activate

# Set cache directories to scratch for better I/O
export HF_HOME=$SCRATCH/.cache/huggingface
export TORCH_HOME=$SCRATCH/.cache/torch
export MNE_DATA=$SCRATCH/mne_data

# Configure for optimal A100 performance
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_BENCHMARK=1
export OMP_NUM_THREADS=8

# Change to project directory
cd $HOME/eeg_challenge

# Create log directory
mkdir -p logs

# Run training with A100-optimized settings
python train_s3_extended.py \
    --pretrain_epochs 200 \
    --finetune_epochs 100 \
    --batch_size 128 \
    --max_subjects 500 \
    --use_subject_invariant \
    --save_every 10 \
    --hidden_dim 128 \
    --num_workers 8 \
    --checkpoint_dir $SCRATCH/checkpoints/eeg_s3_${SLURM_JOB_ID} \
    --experiment_name greene_a100_run_${SLURM_JOB_ID}

# Copy results back to home
cp -r $SCRATCH/checkpoints/eeg_s3_${SLURM_JOB_ID} $HOME/eeg_challenge/results/