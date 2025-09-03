#!/usr/bin/env python3
"""
Extended training script for full HBN S3 dataset
Trains for 100+ epochs with proper checkpointing and monitoring
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.enn_eeg_model import EEGEANN
from models.subject_invariant import SubjectInvariantEEGENN
from utils.training import TransferLearningTrainer
from data.s3_data_loader import create_s3_data_loaders


def parse_args():
    parser = argparse.ArgumentParser(description='Extended S3 EEG Training')
    
    # Data parameters
    parser.add_argument('--s3_bucket', default='fcp-indi', help='S3 bucket name')
    parser.add_argument('--data_prefix', default='R5', help='Data release prefix')
    parser.add_argument('--max_subjects', type=int, default=None, help='Max subjects (None for all)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Data loader workers')
    
    # Model parameters
    parser.add_argument('--model_type', choices=['enn', 'subject_invariant'], default='enn')
    parser.add_argument('--n_channels', type=int, default=129, help='Number of EEG channels')
    parser.add_argument('--n_times', type=int, default=200, help='Time samples per window')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimensions')
    parser.add_argument('--n_filters', type=int, default=40, help='Number of CNN filters')
    
    # Training parameters - EXTENDED
    parser.add_argument('--pretrain_epochs', type=int, default=100, help='Pretraining epochs')
    parser.add_argument('--finetune_epochs', type=int, default=50, help='Fine-tuning epochs')
    parser.add_argument('--pretrain_lr', type=float, default=1e-3, help='Pretrain learning rate')
    parser.add_argument('--finetune_lr', type=float, default=1e-4, help='Finetune learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    
    # Subject-invariant parameters
    parser.add_argument('--use_subject_invariant', action='store_true', help='Use subject invariant training')
    parser.add_argument('--lambda_reversal', type=float, default=0.1, help='Gradient reversal lambda')
    parser.add_argument('--n_subjects', type=int, default=1000, help='Expected number of subjects')
    
    # Infrastructure
    parser.add_argument('--checkpoint_dir', default='./checkpoints_s3_extended', help='Checkpoint directory')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--experiment_name', default='s3_extended_training', help='Experiment name')
    
    return parser.parse_args()


def create_model(args, device):
    """Create the EEG model"""
    # First create base EEG model
    base_model = EEGEANN(
        n_chans=args.n_channels,
        n_times=args.n_times,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        n_filters=args.n_filters
    )
    
    if args.use_subject_invariant:
        # Wrap base model with subject-invariant training
        model = SubjectInvariantEEGENN(
            base_model=base_model,
            n_subjects=args.n_subjects,
            feature_dim=args.hidden_dim,
            lambda_reversal=args.lambda_reversal
        )
    else:
        model = base_model
    
    return model.to(device)


def main():
    args = parse_args()
    
    print("=" * 60)
    print("🚀 EXTENDED S3 EEG TRAINING")
    print("=" * 60)
    print(f"S3 Bucket: {args.s3_bucket}")
    print(f"Data Prefix: {args.data_prefix}")
    print(f"Max Subjects: {args.max_subjects or 'ALL'}")
    print(f"Pretrain Epochs: {args.pretrain_epochs}")
    print(f"Finetune Epochs: {args.finetune_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Model Type: {args.model_type}")
    print(f"Subject Invariant: {args.use_subject_invariant}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create data loaders
    print("\n🔄 Creating S3 data loaders...")
    start_time = time.time()
    
    try:
        data_loaders = create_s3_data_loaders(
            bucket=args.s3_bucket,
            prefix=args.data_prefix,
            batch_size=args.batch_size,
            num_workers=min(args.num_workers, 2),  # Limit for S3
            max_subjects=args.max_subjects
        )
        
        data_load_time = time.time() - start_time
        print(f"✅ Data loaders created in {data_load_time:.1f}s")
        
        # Print dataset sizes
        for name, loader in data_loaders.items():
            print(f"  {name}: {len(loader)} batches ({len(loader.dataset)} samples)")
            
    except Exception as e:
        print(f"❌ Failed to create S3 data loaders: {e}")
        print("Falling back to mock data for development...")
        
        # Create mock data loaders for development
        from torch.utils.data import TensorDataset
        
        n_samples = 1000
        X = torch.randn(n_samples, args.n_channels, args.n_times)
        y = torch.randn(n_samples, 1)
        
        dataset = TensorDataset(X, y)
        data_loaders = {
            'sus_train': DataLoader(dataset, batch_size=args.batch_size, shuffle=True),
            'sus_valid': DataLoader(dataset, batch_size=args.batch_size),
            'ccd_train': DataLoader(dataset, batch_size=args.batch_size, shuffle=True),
            'ccd_valid': DataLoader(dataset, batch_size=args.batch_size),
        }
        print("Using mock data loaders")
    
    # Create model
    print(f"\n🧠 Creating {args.model_type} model...")
    model = create_model(args, device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Create trainer
    print("\n🏋️ Setting up trainer...")
    trainer = TransferLearningTrainer(
        model=model,
        device=device,
        use_wandb=False,
        experiment_name=args.experiment_name
    )
    
    # Save configuration
    config = vars(args)
    config_path = os.path.join(args.checkpoint_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")
    
    # Phase 1: Extended Pretraining on SuS data
    print("\n" + "=" * 60)
    print("PHASE 1: EXTENDED PRETRAINING ON SURROUND SUPPRESSION")
    print("=" * 60)
    
    if 'sus_train' in data_loaders and 'sus_valid' in data_loaders:
        pretrain_history = trainer.pretrain_passive(
            train_loader=data_loaders['sus_train'],
            valid_loader=data_loaders['sus_valid'],
            epochs=args.pretrain_epochs,
            lr=args.pretrain_lr,
            weight_decay=args.weight_decay,
            patience=args.patience
        )
        print("✅ Pretraining completed!")
    else:
        print("❌ No SuS data available for pretraining")
        return
    
    # Phase 2: Extended Fine-tuning on CCD data  
    print("\n" + "=" * 60)
    print("PHASE 2: EXTENDED FINE-TUNING ON CONTRAST CHANGE DETECTION")
    print("=" * 60)
    
    if 'ccd_train' in data_loaders and 'ccd_valid' in data_loaders:
        finetune_history = trainer.finetune_active(
            train_loader=data_loaders['ccd_train'],
            valid_loader=data_loaders['ccd_valid'],
            epochs=args.finetune_epochs,
            lr=args.finetune_lr,
            weight_decay=args.weight_decay,
            freeze_backbone=False,  # Don't freeze for ENN
            patience=args.patience
        )
        print("✅ Fine-tuning completed!")
    else:
        print("❌ No CCD data available for fine-tuning")
        return
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    test_metrics = {}
    
    # Test on CCD (primary task)
    if 'ccd_test' in data_loaders:
        print("Testing on CCD task...")
        ccd_metrics = trainer.evaluate(data_loaders['ccd_test'])
        test_metrics['ccd'] = ccd_metrics
        print(f"CCD Test - MAE: {ccd_metrics['mae']:.4f}, R²: {ccd_metrics['r2']:.4f}")
        if 'auc' in ccd_metrics:
            print(f"CCD Test - AUC: {ccd_metrics['auc']:.4f}")
    
    # Test on SuS (transfer evaluation)
    if 'sus_test' in data_loaders:
        print("Testing on SuS task...")
        sus_metrics = trainer.evaluate(data_loaders['sus_test'])
        test_metrics['sus'] = sus_metrics  
        print(f"SuS Test - MAE: {sus_metrics['mae']:.4f}, R²: {sus_metrics['r2']:.4f}")
    
    # Save final results
    results = {
        'config': config,
        'pretrain_history': pretrain_history,
        'finetune_history': finetune_history,
        'test_metrics': test_metrics
    }
    
    results_path = os.path.join(args.checkpoint_dir, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 Training completed! Results saved to {results_path}")
    
    # Print summary
    if 'ccd' in test_metrics:
        ccd = test_metrics['ccd']
        print(f"\n📊 FINAL PERFORMANCE SUMMARY:")
        print(f"CCD MAE: {ccd['mae']:.4f} seconds")
        print(f"CCD R²: {ccd['r2']:.4f}")
        if 'auc' in ccd:
            print(f"CCD AUC: {ccd['auc']:.4f}")
        
        # Performance assessment
        if ccd['mae'] < 0.5:
            print("🏆 EXCELLENT performance!")
        elif ccd['mae'] < 0.7:
            print("✅ GOOD performance")
        elif ccd['mae'] < 1.0:
            print("⚠️ MODERATE performance")
        else:
            print("❌ POOR performance - needs improvement")


if __name__ == "__main__":
    main()