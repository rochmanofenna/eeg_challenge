#!/usr/bin/env python3
"""
B200-Optimized Full Dataset Training
Designed for 2x B200 GPUs (192GB each) - Full 3,326 file dataset
Target: 30-45 minutes total training time
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Environment setup for B200s
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

def main():
    parser = argparse.ArgumentParser(description='B200 Full Dataset Training')
    
    # B200-optimized parameters
    parser.add_argument('--batch_size', type=int, default=2048, help='Total batch size across GPUs')
    parser.add_argument('--pretrain_epochs', type=int, default=80, help='Pretrain epochs')
    parser.add_argument('--finetune_epochs', type=int, default=30, help='Finetune epochs')
    parser.add_argument('--lr', type=float, default=0.008, help='Base learning rate')
    parser.add_argument('--num_workers', type=int, default=16, help='Data workers per GPU')
    
    args = parser.parse_args()
    
    print("🚀 B200 FULL DATASET TRAINING")
    print("=" * 50)
    print(f"GPUs: {torch.cuda.device_count()} B200s detected")
    print(f"Batch Size: {args.batch_size}")
    print(f"Total Epochs: {args.pretrain_epochs + args.finetune_epochs}")
    print("=" * 50)
    
    # Import after argument parsing to avoid long startup
    from models.enn_eeg_model import EEGEANN
    from data.s3_data_loader import S3EEGDataset
    from utils.training import TransferLearningTrainer
    
    # Create model
    model = EEGEANN(
        n_chans=64,
        n_times=200,
        output_dim=1,
        hidden_dim=128
    )
    
    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.cuda()
    
    # B200-optimized dataset
    print("Loading S3 dataset (3,326 files)...")
    train_dataset = S3EEGDataset(
        task='contrast_change_detection',
        max_subjects=None,  # Use ALL subjects
        preprocessing_steps=['bandpass', 'normalize'],
        cache_processed=True
    )
    
    # B200-optimized dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers * torch.cuda.device_count(),
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    
    print(f"Dataset size: {len(train_dataset)} samples")
    print(f"Steps per epoch: {len(train_loader)}")
    
    # B200-optimized optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr * torch.cuda.device_count(),  # Scale LR with GPUs
        weight_decay=1e-4,
        betas=(0.9, 0.95)  # More aggressive for B200
    )
    
    # Cosine annealing for smooth convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.pretrain_epochs + args.finetune_epochs
    )
    
    # Create trainer
    trainer = TransferLearningTrainer(
        model=model,
        device='cuda',
        checkpoint_dir='checkpoints'
    )
    
    # Training
    start_time = time.time()
    
    print("\n📊 Starting Pretraining Phase...")
    trainer.pretrain(
        train_loader,
        optimizer,
        n_epochs=args.pretrain_epochs,
        val_loader=None,  # Skip validation for speed
        scheduler=scheduler,
        use_amp=True  # Mixed precision for B200
    )
    
    print("\n🎯 Starting Finetuning Phase...")  
    trainer.finetune(
        train_loader,
        optimizer,
        n_epochs=args.finetune_epochs,
        val_loader=None,
        scheduler=scheduler,
        use_amp=True
    )
    
    # Save final model
    total_time = (time.time() - start_time) / 60
    print(f"\n✅ Training Complete! Time: {total_time:.1f} minutes")
    
    # Save model
    save_path = 'checkpoints/b200_full_dataset_final.pt'
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
        
    torch.save({
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'training_time_minutes': total_time,
        'dataset_size': len(train_dataset)
    }, save_path)
    
    print(f"Model saved to: {save_path}")
    print(f"Expected performance: MAE < 0.5, R² > 0.3")

if __name__ == '__main__':
    main()