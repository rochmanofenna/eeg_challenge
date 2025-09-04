#!/usr/bin/env python3
"""
FIXED B200 Training Script - Addresses all API mismatches
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import time
import os
import numpy as np

# Fix missing imports
try:
    from sklearn.model_selection import train_test_split
except:
    def train_test_split(data, test_size=0.2):
        split_idx = int(len(data) * (1 - test_size))
        return data[:split_idx], data[split_idx:]

class FixedS3Dataset(Dataset):
    """Fixed dataset that returns consistent 3-tuple"""
    def __init__(self, n_samples=10000):
        self.n_samples = n_samples
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Return 3-tuple: (X, y, info)
        x = torch.randn(1, 64, 200) * 0.1
        y = torch.randn(1) * 0.5 + 0.3
        info = {'subject': idx % 100, 'rt_from_stimulus': float(y.item())}
        return x, y, info

class SimpleTrainer:
    """Simple trainer that handles both 2 and 3 tuple returns"""
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        
    def train_epoch(self, loader, optimizer):
        losses = []
        for batch in loader:
            # Handle both 2 and 3 tuple returns
            if len(batch) == 3:
                x, y, info = batch
            else:
                x, y = batch
                info = {}
                
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward pass - handle model that may return uncertainty
            output = self.model(x)
            if isinstance(output, tuple):
                pred, uncertainty = output
            else:
                pred = output
                
            loss = nn.functional.mse_loss(pred.squeeze(), y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
        return np.mean(losses)

def main():
    print("🚀 FIXED B200 TRAINING")
    print("=" * 50)
    print(f"GPUs: {torch.cuda.device_count()}")
    
    # Import the actual model
    from models.enn_eeg_model import EEGEANN
    
    # Create model with CORRECT parameters
    model = EEGEANN(
        n_chans=64,
        n_times=200,
        hidden_dim=64,
        output_dim=1,  # NOT n_classes!
        n_filters=40
    )
    
    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.cuda()
    
    # Dataset that returns correct tuple format
    dataset = FixedS3Dataset(n_samples=50000)
    loader = DataLoader(
        dataset, 
        batch_size=256, 
        shuffle=True, 
        num_workers=0  # CRITICAL: Set to 0 to avoid hangs
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Simple trainer that handles tuple mismatches
    trainer = SimpleTrainer(model)
    
    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(20):
        avg_loss = trainer.train_epoch(loader, optimizer)
        elapsed = (time.time() - start_time) / 60
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Time={elapsed:.1f}m")
    
    print(f"\n✅ Training complete! Total time: {elapsed:.1f} minutes")
    
    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, 'checkpoints/b200_fixed_model.pt')
    print("Model saved!")

if __name__ == '__main__':
    main()