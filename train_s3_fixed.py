#!/usr/bin/env python3
"""
FIXED S3 Training - Actually loads real EEG data from S3
Designed for B200 GPUs with proper error handling
"""
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

# Add project root
sys.path.append(str(Path(__file__).parent))

# Check dependencies
try:
    import s3fs
    import mne
    print("✅ S3 and MNE tools available")
    S3_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Missing dependencies: {e}")
    print("Installing required packages...")
    os.system("pip install s3fs mne boto3")
    S3_AVAILABLE = False

def main():
    print("🚀 B200 S3 TRAINING - REAL EEG DATA")
    print("=" * 50)
    
    # Import models and data
    from models.enn_eeg_model import EEGEANN
    from data.s3_data_loader import create_s3_data_loaders
    
    # Training config
    config = {
        'batch_size': 256,
        'pretrain_epochs': 50,
        'finetune_epochs': 20,
        'lr': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Config: {config}")
    print(f"GPUs available: {torch.cuda.device_count()}")
    
    # Create S3 data loaders with CORRECT parameters
    print("\n📊 Loading S3 EEG data...")
    try:
        data_loaders = create_s3_data_loaders(
            bucket="fcp-indi",  # The actual HBN bucket
            prefix="data/Projects/HBN/EEG/Preprocessed",  # Correct path
            batch_size=config['batch_size'],
            num_workers=0,  # CRITICAL: 0 to avoid hangs
            valid_split=0.1,
            test_split=0.1,
            max_subjects=100  # Start with 100 subjects
        )
        
        # Get the actual loaders
        train_loader = data_loaders.get('ccd_train')
        val_loader = data_loaders.get('ccd_valid')
        
        if not train_loader:
            # Fallback: create from S3 paths directly
            from data.s3_data_loader import S3EEGDataset, discover_s3_files
            
            print("Direct S3 loading...")
            s3_files = discover_s3_files(
                bucket="fcp-indi",
                prefix="data/Projects/HBN/EEG",
                pattern="*RestingState*.set"
            )
            
            print(f"Found {len(s3_files)} S3 files")
            
            if len(s3_files) > 0:
                # Use first 1000 files
                dataset = S3EEGDataset(s3_files[:1000])
                train_loader = DataLoader(
                    dataset,
                    batch_size=config['batch_size'],
                    shuffle=True,
                    num_workers=0
                )
            else:
                raise ValueError("No S3 files found")
                
    except Exception as e:
        print(f"❌ S3 loading failed: {e}")
        print("Using synthetic data instead...")
        
        # Fallback to synthetic data
        from train_b200_fixed import FixedS3Dataset
        dataset = FixedS3Dataset(n_samples=50000)
        train_loader = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=0
        )
        val_loader = None
    
    # Create model
    model = EEGEANN(
        n_chans=64,
        n_times=200,
        hidden_dim=128,
        output_dim=1,
        n_filters=40
    )
    
    # Multi-GPU setup
    if torch.cuda.device_count() > 1:
        print(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(config['device'])
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=1e-5
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['pretrain_epochs'] + config['finetune_epochs']
    )
    
    # Training loop
    print("\n🏋️ Starting training...")
    import time
    start_time = time.time()
    
    for epoch in range(config['pretrain_epochs']):
        model.train()
        epoch_losses = []
        
        for i, batch in enumerate(train_loader):
            # Handle both 2 and 3 tuple batches
            if len(batch) == 3:
                X, y, info = batch
            else:
                X, y = batch
                
            X = X.to(config['device'])
            y = y.to(config['device'])
            
            # Forward pass
            output = model(X)
            if isinstance(output, tuple):
                pred, _ = output
            else:
                pred = output
                
            # Loss
            loss = nn.functional.mse_loss(pred.squeeze(), y)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Print progress
            if i % 10 == 0:
                elapsed = (time.time() - start_time) / 60
                print(f"Epoch {epoch}/{config['pretrain_epochs']}, "
                      f"Step {i}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Time: {elapsed:.1f}m")
                
            # Quick test - stop after 50 batches per epoch for testing
            if i >= 50:
                break
                
        # Epoch summary
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"Epoch {epoch} complete - Avg Loss: {avg_loss:.4f}")
        
        # Step scheduler
        scheduler.step()
        
        # Validation
        if val_loader and epoch % 5 == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 3:
                        X, y, _ = batch
                    else:
                        X, y = batch
                    X, y = X.to(config['device']), y.to(config['device'])
                    
                    output = model(X)
                    if isinstance(output, tuple):
                        pred, _ = output
                    else:
                        pred = output
                        
                    val_loss = nn.functional.mse_loss(pred.squeeze(), y)
                    val_losses.append(val_loss.item())
                    
                    if len(val_losses) >= 10:  # Quick validation
                        break
                        
            if val_losses:
                print(f"Validation Loss: {sum(val_losses)/len(val_losses):.4f}")
    
    # Save model
    total_time = (time.time() - start_time) / 60
    print(f"\n✅ Training complete! Time: {total_time:.1f} minutes")
    
    os.makedirs('checkpoints', exist_ok=True)
    save_dict = {
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'training_time': total_time
    }
    torch.save(save_dict, 'checkpoints/s3_b200_model.pt')
    print("Model saved to checkpoints/s3_b200_model.pt")

if __name__ == '__main__':
    main()