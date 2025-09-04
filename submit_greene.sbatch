#!/bin/bash
#SBATCH --job-name=eeg_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1              # Request 1 GPU (V100/A100/RTX8000)
#SBATCH --time=04:00:00           # 4 hours
#SBATCH --output=/scratch/rr3758/eeg_challenge/logs/eeg_%j.out
#SBATCH --mail-user=rr3758@nyu.edu
#SBATCH --mail-type=END,FAIL

# Create logs directory if it doesn't exist
mkdir -p /scratch/rr3758/eeg_challenge/logs

module purge

# Check which GPU we got
echo "Job started on $(hostname) at $(date)"
nvidia-smi

# Option 1: Use Singularity (RECOMMENDED)
singularity exec --nv \
    --overlay /scratch/rr3758/eeg_challenge/overlay-50GB-500K.ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
source /ext3/miniconda3/etc/profile.d/conda.sh
conda activate eeg
cd /scratch/rr3758/eeg_challenge
python train_greene.py --batch_size 256 --epochs 50
"

# Option 2: Use module system (ALTERNATIVE - uncomment if not using Singularity)
# module load python/3.9.16
# module load cuda/11.8.86
# module load cudnn/8.7.0.84-cuda11
# 
# cd /scratch/rr3758/eeg_challenge
# source venv/bin/activate
# python train_greene.py --batch_size 256 --epochs 50

echo "Job finished at $(date)"