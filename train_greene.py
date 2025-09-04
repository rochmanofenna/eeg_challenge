#!/usr/bin/env python3
"""
Greene HPC Training Script - Optimized for available GPUs
Works with V100, A100, or RTX8000
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import time
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    args = parser.parse_args()
    
    print("="*50)
    print("🎓 NYU GREENE HPC TRAINING")
    print("="*50)
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        print(f"✅ GPU: {gpu_name} x {gpu_count}")
    else:
        print("❌ No GPU available, using CPU")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Import model
    from models.enn_eeg_model import EEGEANN
    
    # Create model
    model = EEGEANN(
        n_chans=64,
        n_times=200,
        hidden_dim=128,
        output_dim=1,
        n_filters=40
    )
    
    # Multi-GPU if available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Try to load S3 data, fallback to synthetic
    try:
        from data.s3_data_loader import S3EEGDataset, discover_s3_files
        print("\n📊 Loading S3 EEG data...")
        
        # Discover S3 files
        s3_files = discover_s3_files(
            bucket="fcp-indi",
            prefix="data/Projects/HBN/EEG",
            pattern="*RestingState*.set",
            max_files=1000  # Limit for testing
        )
        
        if len(s3_files) > 0:
            print(f"Found {len(s3_files)} S3 files")
            dataset = S3EEGDataset(s3_files)
        else:
            raise ValueError("No S3 files found")
            
    except Exception as e:
        print(f"⚠️ S3 loading failed: {e}")
        print("Using synthetic data...")
        
        # Fallback to synthetic
        from train_b200_fixed import FixedS3Dataset
        dataset = FixedS3Dataset(n_samples=10000)
    
    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(loader)}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print("\n🏋️ Starting training...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []
        
        for i, batch in enumerate(loader):
            # Handle 2 or 3 tuple batches
            if len(batch) == 3:
                X, y, info = batch
            else:
                X, y = batch
                
            X, y = X.to(device), y.to(device)
            
            # Forward
            output = model(X)
            if isinstance(output, tuple):
                pred, _ = output
            else:
                pred = output
                
            loss = nn.functional.mse_loss(pred.squeeze(), y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Print progress
            if i % 20 == 0:
                elapsed = (time.time() - start_time) / 60
                print(f"Epoch {epoch+1}/{args.epochs}, Step {i}/{len(loader)}, "
                      f"Loss: {loss.item():.4f}, Time: {elapsed:.1f}m")
        
        # Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
        scheduler.step()
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_dir = '/scratch/' + os.environ.get('USER', 'rr3758') + '/eeg_challenge/checkpoints'
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss
            }
            
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"✅ Saved checkpoint: {checkpoint_path}")
    
    # Final save
    total_time = (time.time() - start_time) / 60
    print(f"\n✅ Training complete! Total time: {total_time:.1f} minutes")
    
    final_path = os.path.join(checkpoint_dir, 'final_model.pt')
    torch.save(checkpoint, final_path)
    print(f"Final model saved: {final_path}")

if __name__ == '__main__':
    main()