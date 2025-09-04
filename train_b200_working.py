#!/usr/bin/env python3
"""
Simplified B200 training that actually works
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time
import os

print("🚀 B200 TRAINING - SIMPLIFIED BUT WORKING")
print("=" * 50)

# Import models
from models.enn_eeg_model import EEGEANN
from data.s3_data_loader import get_hbn_s3_paths, S3EEGDataset

# Get S3 paths
print("Getting S3 file paths...")
s3_paths = get_hbn_s3_paths(max_files=3326)
print(f"Found {len(s3_paths)} S3 files")

# Create dataset
print("Creating dataset...")
dataset = S3EEGDataset(s3_paths[:1000])  # Start with 1000 files

# Create model  
model = EEGEANN(n_chans=64, n_times=200, output_dim=1, hidden_dim=128)

# Multi-GPU
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
model = model.cuda()

# Dataloader
loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Training loop
print("\nStarting training...")
start_time = time.time()

for epoch in range(10):
    for i, (data, target) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        
        # Forward
        output, uncertainty = model(data)
        loss = nn.functional.mse_loss(output.squeeze(), target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")
    
    elapsed = (time.time() - start_time) / 60
    print(f"Epoch {epoch} complete. Time: {elapsed:.1f} minutes")

print(f"\n✅ Training complete! Total time: {elapsed:.1f} minutes")

# Save
os.makedirs('checkpoints', exist_ok=True)
torch.save(model.state_dict(), 'checkpoints/b200_model.pt')