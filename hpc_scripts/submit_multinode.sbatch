#!/bin/bash
#SBATCH --job-name=eeg_multinode
#SBATCH --output=logs/eeg_multi_%j.out
#SBATCH --error=logs/eeg_multi_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --gres=gpu:a100:4
#SBATCH --partition=gpu
#SBATCH --exclusive
#SBATCH --mail-type=ALL

# For multi-node training on 16 A100s (4 nodes x 4 GPUs)

module purge
module load cuda/11.8
module load cudnn/8.6.0
module load python/3.10
module load openmpi/4.1.4

source ~/envs/eeg_env/bin/activate

# Get master node
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# Launch distributed training
srun python hpc_scripts/train_distributed.py \
    --pretrain_epochs 200 \
    --finetune_epochs 100 \
    --batch_size 32 \
    --max_subjects 3000 \
    --use_subject_invariant \
    --save_every 5 \
    --hidden_dim 256 \
    --n_filters 80 \
    --num_workers 8 \
    --checkpoint_dir $SCRATCH/checkpoints/eeg_multi_${SLURM_JOB_ID}