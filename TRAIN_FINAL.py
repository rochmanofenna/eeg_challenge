#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from bef_eeg.pipeline import BEF_EEG
import os

print('\nFINAL TRAINING SCRIPT')
print('='*50)

device = torch.device('cuda:0')
torch.cuda.empty_cache()

# Create learnable data
print('Creating dataset with patterns...')
n_samples = 5000
X = torch.zeros(n_samples, 129, 200)
y = torch.zeros(n_samples)

for i in range(n_samples):
    # Class-specific patterns
    if i % 2 == 0:
        X[i, :10, :] = torch.randn(10, 200) * 0.5 + 0.5  # Higher mean for class 0
        y[i] = 0
    else:
        X[i, :10, :] = torch.randn(10, 200) * 0.5 - 0.5  # Lower mean for class 1
        y[i] = 1
    X[i, 10:, :] = torch.randn(119, 200) * 0.1  # Noise in other channels

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=8, shuffle=True)  # Small batch to fit memory

print(f'Dataset: {len(dataset)} samples')

# Model with reduced complexity
print('\nInitializing BEF model...')
model = BEF_EEG(
    in_chans=129,
    sfreq=100,
    n_paths=8,  # Reduced paths
    K=4,        # Reduced states
    output_dim=1,
    gnn_layers=2,
    dropout=0.1
)

# Try loading checkpoint
checkpoint_path = 'bef_eeg/weights_challenge_1.pt'
if os.path.exists(checkpoint_path):
    try:
        # Load just the compatible weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_dict = model.state_dict()
        
        # Filter out incompatible keys
        compatible = {k: v for k, v in checkpoint.items() 
                     if k in model_dict and v.shape == model_dict[k].shape}
        
        model_dict.update(compatible)
        model.load_state_dict(model_dict, strict=False)
        print(f'Loaded {len(compatible)}/{len(checkpoint)} weights from checkpoint')
    except:
        print('Starting fresh')
else:
    print('No checkpoint found')

model = model.to(device)
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')

# Training
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

print('\nTRAINING')
print('='*50)

for epoch in range(10):
    model.train()
    epoch_losses = []
    correct = 0
    total = 0
    
    for batch_idx, (X_batch, y_batch) in enumerate(loader):
        if batch_idx >= 100:  # Limit batches per epoch
            break
            
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward with memory management
        try:
            output = model(X_batch)
            if isinstance(output, dict):
                output = output['prediction']
            output = output.squeeze()
            
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            
            # Metrics
            with torch.no_grad():
                probs = torch.sigmoid(output)
                preds = (probs > 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
                epoch_losses.append(loss.item())
            
            if batch_idx % 20 == 0:
                print(f'  Batch {batch_idx}: Loss={loss.item():.4f}')
                
        except torch.cuda.OutOfMemoryError:
            print('  OOM - skipping batch')
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            continue
    
    avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
    accuracy = correct / total if total > 0 else 0
    
    print(f'Epoch {epoch+1}/10: Loss={avg_loss:.4f}, Acc={accuracy:.4f}')
    
    # Save checkpoint
    if epoch % 3 == 0:
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pt')
        print(f'  Saved checkpoint')

# Final save
torch.save(model.state_dict(), 'final_trained_model.pt')
print('\nTraining complete! Model saved to final_trained_model.pt')
