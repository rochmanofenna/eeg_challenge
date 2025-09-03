#!/usr/bin/env python3
"""
Distributed training script for multi-GPU HPC clusters
Scales EEG training across multiple A100s using PyTorch DDP
"""

import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import sys

# Add project to path
sys.path.append(str(Path(__file__).parent.parent))

from train_s3_extended import create_model, parse_args
from data.s3_data_loader import create_s3_data_loaders
from utils.training import TransferLearningTrainer


def setup_distributed():
    """Initialize distributed training"""
    # Get SLURM variables
    rank = int(os.environ.get('SLURM_PROCID', 0))
    world_size = int(os.environ.get('SLURM_NTASKS', 1))
    
    # Get node list
    node_list = os.environ.get('SLURM_NODELIST', 'localhost')
    
    # Master node
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '29500')
    
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Initialize process group
    dist.init_process_group(
        backend='nccl',
        init_method=f'env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set device
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()


def main():
    # Parse arguments
    args = parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}')
    
    is_main = rank == 0
    
    if is_main:
        print(f"🚀 Distributed Training on {world_size} GPUs")
        print(f"Node: {os.environ.get('SLURM_NODELIST', 'local')}")
        print(f"Job ID: {os.environ.get('SLURM_JOB_ID', 'local')}")
    
    # Scale batch size by number of GPUs
    args.batch_size = args.batch_size * world_size
    
    # Create model
    model = create_model(args, device)
    model = DDP(model, device_ids=[local_rank])
    
    if is_main:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {total_params:,}")
    
    # Create data loaders with DistributedSampler
    data_loaders = create_s3_data_loaders(
        bucket=args.s3_bucket,
        prefix=args.data_prefix,
        batch_size=args.batch_size // world_size,  # Each GPU gets portion
        num_workers=args.num_workers,
        max_subjects=args.max_subjects
    )
    
    # Add DistributedSampler
    for name, loader in data_loaders.items():
        if 'train' in name:
            sampler = DistributedSampler(
                loader.dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
            loader.sampler = sampler
    
    # Create trainer (only main process logs)
    trainer = TransferLearningTrainer(
        model=model,
        device=device,
        use_wandb=args.use_wandb and is_main,
        experiment_name=f"{args.experiment_name}_rank{rank}"
    )
    
    # Training phases
    if is_main:
        print("\n" + "="*60)
        print("PHASE 1: DISTRIBUTED PRETRAINING")
        print("="*60)
    
    # Synchronize before starting
    dist.barrier()
    
    # Pretrain
    pretrain_history = trainer.pretrain_passive(
        train_loader=data_loaders['sus_train'],
        valid_loader=data_loaders['sus_valid'],
        epochs=args.pretrain_epochs,
        lr=args.pretrain_lr,
        weight_decay=args.weight_decay,
        patience=args.patience
    )
    
    if is_main:
        print("\n" + "="*60)
        print("PHASE 2: DISTRIBUTED FINE-TUNING")
        print("="*60)
    
    # Fine-tune
    finetune_history = trainer.finetune_active(
        train_loader=data_loaders['ccd_train'],
        valid_loader=data_loaders['ccd_valid'],
        epochs=args.finetune_epochs,
        lr=args.finetune_lr,
        weight_decay=args.weight_decay,
        freeze_backbone=False,
        patience=args.patience
    )
    
    # Evaluation (only on main)
    if is_main:
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        
        test_metrics = {}
        if 'ccd_test' in data_loaders:
            test_metrics['ccd'] = trainer.evaluate(data_loaders['ccd_test'])
        if 'sus_test' in data_loaders:
            test_metrics['sus'] = trainer.evaluate(data_loaders['sus_test'])
        
        # Save results
        import json
        results = {
            'config': vars(args),
            'world_size': world_size,
            'pretrain_history': pretrain_history,
            'finetune_history': finetune_history,
            'test_metrics': test_metrics
        }
        
        results_path = os.path.join(args.checkpoint_dir, 'final_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n🎉 Distributed training completed! Results: {results_path}")
    
    # Clean up
    cleanup_distributed()


if __name__ == "__main__":
    main()