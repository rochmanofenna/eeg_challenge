#!/usr/bin/env python3
"""
Ultra-fast B200 training script - 30-40 minutes total
Optimized for 2x B200 GPUs (192GB each)
"""
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import os
import time

# Set optimal environment
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['NCCL_DEBUG'] = 'INFO'

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train_b200(rank, world_size):
    print(f"Starting B200 training on GPU {rank}")
    setup_distributed(rank, world_size)
    
    # Ultra-aggressive settings for B200
    config = {
        'pretrain_epochs': 80,      # Reduced but still effective
        'finetune_epochs': 30,      # Reduced for speed
        'batch_size': 2048,         # Massive batches (192GB memory)
        'lr': 0.01,                 # Higher LR for large batches
        'weight_decay': 1e-4,
        'max_subjects': 1000,       # Use more data efficiently
        'num_workers': 16,          # Max data loading
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True,
    }
    
    try:
        # Import after distributed setup
        from models.enn_eeg_model import SubjectInvariantEEGENN
        from data.s3_data_loader import S3EEGDataLoader
        from utils.train_utils import EEGTrainer
        
        # Model setup
        model = SubjectInvariantEEGENN(
            n_chans=64, n_times=200, n_classes=1,
            use_subject_invariant=True,
            dropout_rate=0.2
        ).cuda()
        
        # Wrap in DDP
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
        
        # Aggressive optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['lr'] * world_size,  # Scale LR with GPUs
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.95)  # More aggressive momentum
        )
        
        # Cosine annealing for fast convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['pretrain_epochs'] + config['finetune_epochs']
        )
        
        # Ultra-fast data loader
        train_loader = S3EEGDataLoader(
            batch_size=config['batch_size'] // world_size,
            task='contrast_change_detection',
            max_subjects=config['max_subjects'],
            num_workers=config['num_workers'],
            pin_memory=config['pin_memory'],
            prefetch_factor=config['prefetch_factor'],
            persistent_workers=config['persistent_workers'],
            distributed=True,
            rank=rank,
            world_size=world_size
        )
        
        # Training loop
        trainer = EEGTrainer(model, optimizer, scheduler, rank)
        
        print(f"GPU {rank}: Starting ultra-fast training...")
        start_time = time.time()
        
        # Pretrain phase
        for epoch in range(config['pretrain_epochs']):
            train_loss = trainer.train_epoch(train_loader, epoch, 'pretrain')
            if rank == 0 and epoch % 10 == 0:
                elapsed = (time.time() - start_time) / 60
                print(f"Pretrain Epoch {epoch}: Loss={train_loss:.4f}, Time={elapsed:.1f}m")
        
        # Finetune phase  
        for epoch in range(config['finetune_epochs']):
            train_loss = trainer.train_epoch(train_loader, epoch, 'finetune')
            if rank == 0 and epoch % 5 == 0:
                elapsed = (time.time() - start_time) / 60
                print(f"Finetune Epoch {epoch}: Loss={train_loss:.4f}, Time={elapsed:.1f}m")
        
        total_time = (time.time() - start_time) / 60
        if rank == 0:
            print(f"B200 Training Complete! Total time: {total_time:.1f} minutes")
            
            # Save final model
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'training_time_minutes': total_time
            }, 'checkpoints/b200_final_model.pt')
            
    except Exception as e:
        print(f"GPU {rank} error: {e}")
    finally:
        cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=2, help='Number of B200 GPUs')
    args = parser.parse_args()
    
    print("🚀 B200 Ultra-Fast Training (Target: 30-40 minutes)")
    print("=" * 50)
    
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Launch distributed training
    mp.spawn(train_b200, args=(args.gpus,), nprocs=args.gpus, join=True)
    
    print("✅ B200 training completed!")

if __name__ == '__main__':
    main()