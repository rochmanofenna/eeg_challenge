#!/usr/bin/env python3
"""
Multi-GPU training script for BEF EEG model
Utilizes all 3 available GPUs (nvidia0, nvidia2, nvidia3)
"""

import os
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
import argparse

def setup_multi_gpu():
    """Setup multi-GPU environment"""

    print("=== Multi-GPU Setup ===")

    # GPU environment
    os.environ['NVIDIA_VISIBLE_DEVICES'] = 'all'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'  # All 3 GPUs

    gpu_devices = [f for f in os.listdir('/dev') if f.startswith('nvidia') and f[-1].isdigit()]
    print(f"Available GPU devices: {sorted(gpu_devices)}")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"🚀 PyTorch detected {gpu_count} GPUs")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        return gpu_count
    else:
        print("⚠️  PyTorch CUDA not available - training on CPU")
        return 0

def create_multi_gpu_model(config, gpu_count=0):
    """Create model with multi-GPU support"""

    from train import BEFTrainer
    from bef_eeg.pipeline import PretrainableBEF

    # Determine output dimensions based on task
    # For real HBN data, we're doing binary classification
    task = config.get('task', 'regression').lower()
    if task == 'multiclass' or task == 'binary':
        output_dim = 1  # Binary classification uses 1 output with sigmoid
        config['task'] = 'binary'  # Force binary for real data
    else:
        output_dim = 1

    # Create base model
    model = PretrainableBEF(
        in_chans=config['in_chans'],
        sfreq=config['sfreq'],
        n_paths=config['bicep']['n_paths'],
        K=config['enn']['K'],
        gnn_hidden=config['fusion']['gnn_hid'],
        gnn_layers=config['fusion']['layers'],
        dropout=config['fusion']['dropout'],
        output_dim=output_dim
    )

    device = "cpu"

    if gpu_count > 1:
        print(f"🔥 Setting up DataParallel across {gpu_count} GPUs")
        device = "cuda"
        model = model.to(device)
        model = nn.DataParallel(model, device_ids=list(range(gpu_count)))
        print(f"Model distributed across GPUs: {list(range(gpu_count))}")

    elif gpu_count == 1:
        print("🚀 Using single GPU")
        device = "cuda:0"
        model = model.to(device)

    else:
        print("⚠️  Using CPU")
        device = "cpu"

    return model, device

def train_multi_gpu(config_path='bef_eeg/config.yaml'):
    """Main multi-GPU training function"""

    print("=== Multi-GPU BEF Training ===")

    # Setup GPUs
    gpu_count = setup_multi_gpu()

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Adjust batch size for multi-GPU
    if gpu_count > 1:
        config['train']['batch_size'] = config['train']['batch_size'] * gpu_count
        print(f"Scaled batch size to {config['train']['batch_size']} for {gpu_count} GPUs")

    # Create model
    model, device = create_multi_gpu_model(config, gpu_count)

    # Create trainer
    from train import BEFTrainer
    trainer = BEFTrainer(model, config, device=device, use_wandb=False)

    # Setup optimizers for full training
    trainer.setup_optimizers(stage="finetune_full")

    # Load REAL dataset from S3
    try:
        from real_hbn_loader import RealHBNDataset
        from torch.utils.data import DataLoader
        print("Loading REAL HBN EEG data from S3...")

        # Create REAL datasets - need more subjects for proper split
        # First, create one dataset to scan and cache subjects
        print("Scanning S3 for available subjects...")
        scan_dataset = RealHBNDataset(
            releases=["cmi_bids_R1"],  # One release
            tasks=["RestingState", "DespicableMe"],
            split="train",
            split_ratios=(1.0, 0.0, 0.0),  # All for scanning
            max_subjects=50,  # Scan up to 50 subjects
            cache_dir="/tmp/real_hbn_cache"
        )
        print(f"Found {len(scan_dataset.subjects_data)} subjects with data")

        # Now create proper splits with cached metadata
        train_dataset = RealHBNDataset(
            releases=["cmi_bids_R1"],
            tasks=["RestingState", "DespicableMe"],
            split="train",
            split_ratios=(0.6, 0.2, 0.2),  # 60/20/20 split
            max_subjects=50,
            cache_dir="/tmp/real_hbn_cache"
        )

        val_dataset = RealHBNDataset(
            releases=["cmi_bids_R1"],
            tasks=["RestingState", "DespicableMe"],
            split="val",
            split_ratios=(0.6, 0.2, 0.2),
            max_subjects=50,
            cache_dir="/tmp/real_hbn_cache"
        )

        test_dataset = RealHBNDataset(
            releases=["cmi_bids_R1"],
            tasks=["RestingState", "DespicableMe"],
            split="test",
            split_ratios=(0.6, 0.2, 0.2),
            max_subjects=50,
            cache_dir="/tmp/real_hbn_cache"
        )

        # Create data loaders
        batch_size = config['train']['batch_size'] // max(1, gpu_count)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4 * max(1, gpu_count),
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )

        print(f"REAL Dataset loaded:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Val: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")

        # Update config for binary classification with real data
        if config.get('task') == 'multiclass':
            config['task'] = 'binary'  # Override to binary for real data
            config['num_classes'] = 2
            print("  Task: Binary classification (RestingState vs DespicableMe)")

    except Exception as e:
        print(f"REAL data loading failed: {e}")
        print("Please ensure you have:")
        print("  1. Internet connection for S3 access")
        print("  2. boto3 installed: pip install boto3")
        print("  3. Sufficient disk space in /tmp/real_hbn_cache")
        import traceback
        traceback.print_exc()
        return None

    # Training parameters
    epochs = config['train']['epochs_finetune']
    print(f"Training for {epochs} epochs on {device}")

    # Training loop
    try:
        # Initialize best metric tracking based on task
        task = config.get('task', 'regression').lower()
        if task == 'multiclass':
            best_val_metric = -float('inf')  # For accuracy (higher is better)
        else:
            best_val_metric = float('inf')  # For loss (lower is better)

        patience = 10  # Number of epochs to wait for improvement
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch+1}/{epochs} ===")

            # Train
            train_metrics = trainer.train_epoch(train_loader, epoch, stage="full")
            print(f"Train - Loss: {train_metrics.get('total', 0):.4f}")

            # Validate
            val_metrics = trainer.evaluate(val_loader, stage='val')

            # Get appropriate validation metric based on task
            task = config.get('task', 'regression').lower()
            if task == 'binary':
                val_loss = val_metrics.get('bce', float('inf'))
                print(f"Val - Loss: {val_loss:.4f}, AUC: {val_metrics.get('auc', 0):.4f}")
            elif task == 'multiclass':
                val_acc = val_metrics.get('acc', 0.0)
                val_loss = -val_acc  # Negative for consistent "lower is better" logic
                print(f"Val - Acc: {val_acc:.4f}, F1: {val_metrics.get('f1_macro', 0):.4f}, CE Loss: {val_metrics.get('ce_loss', float('inf')):.4f}")
            else:
                val_loss = val_metrics.get('mse', float('inf'))
                print(f"Val - MSE: {val_loss:.4f}, MAE: {val_metrics.get('mae', 0):.4f}, R2: {val_metrics.get('r2', 0):.4f}")

            # Save best model
            if task == 'multiclass':
                # For multiclass, val_loss is negative accuracy
                is_better = val_loss > best_val_metric
            else:
                is_better = val_loss < best_val_metric

            if is_better:
                best_val_metric = val_loss
                patience_counter = 0  # Reset patience counter
                metric_name = "accuracy" if task == 'multiclass' else "loss"
                print(f"💾 New best model saved ({metric_name}: {abs(val_loss):.4f})")

                # Save checkpoint
                checkpoint_path = "checkpoints/best_model.pt"
                os.makedirs("checkpoints", exist_ok=True)

                if isinstance(trainer.model, nn.DataParallel):
                    torch.save(trainer.model.module.state_dict(), checkpoint_path)
                else:
                    torch.save(trainer.model.state_dict(), checkpoint_path)

                print(f"Model saved to {checkpoint_path}")
            else:
                patience_counter += 1
                print(f"No improvement (patience: {patience_counter}/{patience})")

            # Early stopping with patience
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

        # Final evaluation
        print(f"\n=== Final Evaluation ===")
        test_metrics = trainer.evaluate(test_loader, stage='test')

        print(f"Final Test Results:")
        for key, value in test_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.4f}")

        return test_metrics

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='bef_eeg/config.yaml', help='Config file path')
    args = parser.parse_args()

    print("🚀 Starting Multi-GPU BEF Training")
    print("=" * 50)

    # Load config to check task type
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    results = train_multi_gpu(args.config)

    if results:
        print("\n🎉 Multi-GPU training completed successfully!")

        # Print appropriate final metrics based on task
        task = config.get('task', 'regression').lower()
        if task == 'binary':
            print(f"Final AUC: {results.get('auc', 0):.4f}")
            print(f"Final BCE: {results.get('bce', 0):.4f}")
        elif task == 'multiclass':
            print(f"Final Accuracy: {results.get('acc', 0):.4f}")
            print(f"Final F1 (macro): {results.get('f1_macro', 0):.4f}")
            print(f"Final CE Loss: {results.get('ce_loss', float('inf')):.4f}")
        else:
            print(f"Final MSE: {results.get('mse', 0):.4f}")
            print(f"Final MAE: {results.get('mae', 0):.4f}")
            print(f"Final R2: {results.get('r2', 0):.4f}")
    else:
        print("\n❌ Training failed")