# HPC Deployment Guide for Greene Cluster

## Quick Start

1. **Setup Environment** (one-time):
```bash
bash hpc_scripts/setup_greene_env.sh
```

2. **Copy Code to Greene**:
```bash
# From your local machine
rsync -avz --exclude='*.zip' --exclude='R5_mini*' --exclude='__pycache__' \
    ~/research/eeg_challenge/ \
    your_netid@greene.hpc.nyu.edu:~/eeg_challenge/
```

3. **Submit Single GPU Job**:
```bash
# On Greene
cd ~/eeg_challenge
sbatch hpc_scripts/submit_greene.sbatch
```

4. **Submit Multi-Node Job** (16 A100s):
```bash
sbatch hpc_scripts/submit_multinode.sbatch
```

## Configuration Options

### Single A100 (Recommended Start)
- **Time**: ~10-20 hours for full training
- **Cost**: Free on Greene!
- **Performance**: 10-20x faster than GTX 1650

### Multi-Node (4 nodes × 4 A100s = 16 GPUs)
- **Time**: ~2-4 hours for full training
- **Batch Size**: 512 total (32 per GPU)
- **Process**: All 3,326 subjects

## Monitoring

```bash
# Check job status
squeue -u $USER

# Watch output
tail -f logs/eeg_train_*.out

# Cancel job
scancel <job_id>

# Check GPU usage
ssh <node_name> nvidia-smi
```

## Key Optimizations for A100

1. **Mixed Precision Training**: 
   - Add `--use_amp` flag for 2x speedup
   
2. **Gradient Accumulation**:
   - Simulate larger batches: `--gradient_accumulation_steps 4`

3. **Pin Memory**:
   - Already enabled in data loaders

4. **Tensor Cores**:
   - Dimensions divisible by 8 for optimal performance

## Expected Performance

| Setup | Time/Epoch | Total Time | 
|-------|------------|------------|
| GTX 1650 | ~90 min | 200+ hours |
| 1× A100 | ~5 min | 10-15 hours |
| 4× A100 | ~1.5 min | 3-4 hours |
| 16× A100 | ~25 sec | 1-2 hours |

## Data Management

The S3 data will be cached to `$SCRATCH` for fast access:
- First epoch: Downloads from S3
- Subsequent epochs: Reads from cache

## Results

Results will be saved to:
- Checkpoints: `$SCRATCH/checkpoints/eeg_s3_<job_id>/`
- Final results: `$HOME/eeg_challenge/results/`
- Logs: `$HOME/eeg_challenge/logs/`

## Troubleshooting

1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **S3 Timeout**: Use cached data or increase num_workers
3. **Module Errors**: Check available modules with `module avail`