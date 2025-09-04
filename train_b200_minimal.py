#!/usr/bin/env python3
"""
Minimal B200 training - just get it working
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
import time

class SimpleEEGDataset(Dataset):
    """Synthetic EEG data for testing"""
    def __init__(self, n_samples=10000):
        self.n_samples = n_samples
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Synthetic EEG: (channels=64, time=200)
        x = torch.randn(1, 64, 200) * 0.1
        y = torch.randn(1) * 0.5 + 0.3  # Reaction time
        return x, y

# Simple EEG model
class SimpleEEGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 40, (1, 25))
        self.conv2 = nn.Conv2d(40, 40, (64, 1))
        self.pool = nn.AdaptiveAvgPool2d((1, 10))
        self.fc = nn.Linear(400, 1)
        
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Main training
print("🚀 B200 MINIMAL TRAINING")
print("=" * 50)
print(f"GPUs available: {torch.cuda.device_count()}")

# Model
model = SimpleEEGModel()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model = model.cuda()

# Data
dataset = SimpleEEGDataset(n_samples=50000)
loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=4)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
print("\nStarting training...")
start = time.time()

for epoch in range(20):
    losses = []
    for i, (x, y) in enumerate(loader):
        x, y = x.cuda(), y.cuda()
        
        pred = model(x).squeeze()
        loss = nn.functional.mse_loss(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if i % 20 == 0:
            print(f"Epoch {epoch}, Step {i}/{len(loader)}, Loss: {loss.item():.4f}")
    
    print(f"Epoch {epoch} - Avg Loss: {np.mean(losses):.4f}, Time: {(time.time()-start)/60:.1f}m")

print(f"\n✅ Done! Total time: {(time.time()-start)/60:.1f} minutes")
torch.save(model.state_dict(), 'b200_model.pt')