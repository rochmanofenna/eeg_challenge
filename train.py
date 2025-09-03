"""
Main training script for EEG Challenge
Implements full pipeline: pretraining → fine-tuning → evaluation
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.enn_eeg_model import EEGEANN, MultiTaskEEGENN
from models.subject_invariant import create_subject_invariant_model, MixUpAugmentation
from data.data_utils import EEGAugmentation, SubjectNormalization, create_data_loaders
from utils.training import TransferLearningTrainer

# Import challenge data loading utilities
try:
    from eegdash.dataset import EEGChallengeDataset as EEGDashDataset
    from braindecode.preprocessing import preprocess, Preprocessor
    from braindecode.datasets import create_windows_from_events
    HAVE_CHALLENGE_DATA = True
except ImportError:
    print("Warning: Challenge data libraries not found. Using mock data.")
    HAVE_CHALLENGE_DATA = False


def create_windows_from_raw(dataset, config):
    """Create windows from raw EEG data"""
    # This is simplified - in practice, use the full preprocessing from the challenge notebook
    
    # Define preprocessing
    preprocessors = [
        # Add your preprocessing steps here based on challenge requirements
    ]
    
    if preprocessors:
        preprocess(dataset, preprocessors)
    
    # Create windows
    # This would use the actual windowing logic from the challenge
    # For now, returning dataset as-is
    return dataset


def prepare_data(config):
    """Prepare data loaders for training"""
    
    if config.get('use_s3_data', False):
        # Use S3 streaming for full datasets
        from data.s3_data_loader import create_s3_data_loaders
        
        loaders = create_s3_data_loaders(
            bucket=config.get('s3_bucket', 'hbn-eeg'),
            prefix=config.get('data_release', 'R5'),
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            valid_split=config['valid_split'],
            test_split=config['test_split'],
            max_subjects=config.get('max_subjects', None)
        )
        
    elif not config['use_mock_data']:
        # Use real R5 mini data loading
        from data.real_data_loader import load_real_r5_data
        
        real_data_dir = "/home/ryan/research/eeg_challenge/R5_mini_L100-20250903T052429Z-1-001/R5_mini_L100"
        
        loaders = load_real_r5_data(
            data_dir=real_data_dir,
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            valid_split=config['valid_split'],
            test_split=config['test_split']
        )
        
    else:
        # Create mock data for testing
        print("Using mock data for testing...")
        
        from torch.utils.data import TensorDataset
        
        # Mock data dimensions
        n_samples = 1000
        n_channels = 129
        n_times = 200
        
        # Create random data
        X = torch.randn(n_samples, n_channels, n_times)
        y = torch.randn(n_samples, 1)  # Regression targets
        
        # Create mock info for each sample
        mock_info = [{'subject': i % 20, 'rt_from_stimulus': 1.5 + torch.randn(1).item() * 0.3, 
                     'correct': torch.rand(1).item() > 0.5} for i in range(n_samples)]
        
        # Create wrapper dataset that includes info
        class MockEEGDataset(TensorDataset):
            def __init__(self, X, y, info_list):
                super().__init__(X, y)
                self.info_list = info_list
                
            def __getitem__(self, index):
                X, y = super().__getitem__(index)
                info = self.info_list[index]
                return X, y, info
        
        # Create datasets
        dataset = MockEEGDataset(X, y, mock_info)
        
        # Split into train/valid/test
        train_size = int(0.7 * n_samples)
        valid_size = int(0.15 * n_samples)
        test_size = n_samples - train_size - valid_size
        
        # Split indices
        indices = list(range(n_samples))
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:train_size + valid_size]
        test_indices = indices[train_size + valid_size:]
        
        # Create proper subset datasets that maintain the info structure
        class MockEEGSubset:
            def __init__(self, parent_dataset, indices):
                self.parent_dataset = parent_dataset
                self.indices = indices
                
            def __len__(self):
                return len(self.indices)
                
            def __getitem__(self, idx):
                return self.parent_dataset[self.indices[idx]]
        
        train_dataset = MockEEGSubset(dataset, train_indices)
        valid_dataset = MockEEGSubset(dataset, valid_indices)
        test_dataset = MockEEGSubset(dataset, test_indices)
        
        # Create loaders
        loaders = {
            'sus_train': DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0),
            'sus_valid': DataLoader(valid_dataset, batch_size=config['batch_size'], num_workers=0),
            'sus_test': DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=0),
            'ccd_train': DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0),
            'ccd_valid': DataLoader(valid_dataset, batch_size=config['batch_size'], num_workers=0),
            'ccd_test': DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=0),
        }
        
    return loaders


def create_model(config):
    """Create model based on configuration"""
    
    # Base model
    if config['model_type'] == 'enn':
        model = EEGEANN(
            n_chans=config['n_channels'],
            n_times=config['n_times'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            n_filters=config['n_filters']
        )
    elif config['model_type'] == 'multitask':
        model = MultiTaskEEGENN(
            n_chans=config['n_channels'],
            n_times=config['n_times'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            n_filters=config['n_filters'],
            n_psychopathology_factors=4
        )
    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")
        
    # Add subject-invariant wrapper if requested
    if config['use_subject_invariant']:
        model = create_subject_invariant_model(
            base_model=model,
            n_subjects=config['n_subjects'],
            feature_dim=config['feature_dim'],
            lambda_reversal=config['lambda_reversal'],
            use_instance_norm=config['use_instance_norm']
        )
        
    return model


def train_model(model, loaders, config):
    """Train the model using transfer learning strategy"""
    
    # Create trainer
    trainer = TransferLearningTrainer(
        model=model,
        use_wandb=config['use_wandb'],
        experiment_name=config['experiment_name']
    )
    
    # Phase 1: Pretrain on passive task (SuS)
    print("\n" + "="*50)
    print("Phase 1: Pretraining on Surround Suppression")
    print("="*50)
    
    pretrain_history = trainer.pretrain_passive(
        train_loader=loaders['sus_train'],
        valid_loader=loaders['sus_valid'],
        epochs=config['pretrain_epochs'],
        lr=config['pretrain_lr'],
        weight_decay=config['weight_decay'],
        patience=config['patience'],
        self_supervised=config['use_self_supervised']
    )
    
    # Save pretrained model
    torch.save(
        trainer.model.state_dict(),
        os.path.join(config['checkpoint_dir'], 'pretrained_model.pt')
    )
    
    # Phase 2: Fine-tune on active task (CCD)
    print("\n" + "="*50)
    print("Phase 2: Fine-tuning on Contrast Change Detection")
    print("="*50)
    
    finetune_history = trainer.finetune_active(
        train_loader=loaders['ccd_train'],
        valid_loader=loaders['ccd_valid'],
        epochs=config['finetune_epochs'],
        lr=config['finetune_lr'],
        weight_decay=config['weight_decay'],
        freeze_backbone=config['freeze_backbone'],
        patience=config['patience']
    )
    
    # Save fine-tuned model
    torch.save(
        trainer.model.state_dict(),
        os.path.join(config['checkpoint_dir'], 'finetuned_model.pt')
    )
    
    return trainer, pretrain_history, finetune_history


def evaluate_model(trainer, loaders, config):
    """Evaluate model on test sets"""
    
    print("\n" + "="*50)
    print("Final Evaluation")
    print("="*50)
    
    # Evaluate on CCD test set
    ccd_metrics = trainer._validate_multitask(
        loaders['ccd_test'],
        phase="test"
    )
    
    print("\nCCD Test Results:")
    print(f"  MAE: {ccd_metrics['mae']:.4f}")
    print(f"  R²: {ccd_metrics['r2']:.4f}")
    print(f"  AUC: {ccd_metrics['auc']:.4f}")
    print(f"  Balanced Accuracy: {ccd_metrics['balanced_acc']:.4f}")
    
    # Evaluate on SuS test set (transfer performance)
    # Use the same criterion that was used in training
    if hasattr(trainer.model, 'enn_cell'):
        from utils.training import GaussianNLLLoss
        criterion = GaussianNLLLoss()
    else:
        criterion = nn.MSELoss()
        
    sus_metrics = trainer._validate(
        loaders['sus_test'],
        criterion=criterion,
        phase="test"
    )
    
    print("\nSuS Test Results (Transfer):")
    print(f"  MAE: {sus_metrics['mae']:.4f}")
    print(f"  R²: {sus_metrics['r2']:.4f}")
    
    return {'ccd': ccd_metrics, 'sus': sus_metrics}


def prepare_submission(model, config):
    """Prepare submission files for Codabench"""
    
    print("\n" + "="*50)
    print("Preparing Submission")
    print("="*50)
    
    # Extract the core model for submission (without subject-invariant wrapper)
    if hasattr(model, 'base_model'):
        core_model = model.base_model
    else:
        core_model = model
        
    # Save weights for both challenges
    # For Challenge 1: Use the fine-tuned model
    torch.save(
        core_model.state_dict(),
        os.path.join(config['submission_dir'], 'weights_challenge_1.pt')
    )
    
    # For Challenge 2: Could use a separately trained model or same model
    # For now, using the same model
    torch.save(
        core_model.state_dict(),
        os.path.join(config['submission_dir'], 'weights_challenge_2.pt')
    )
    
    # Copy submission.py to submission directory
    import shutil
    shutil.copy(
        'submission.py',
        os.path.join(config['submission_dir'], 'submission.py')
    )
    
    # Create zip file
    import zipfile
    with zipfile.ZipFile(
        os.path.join(config['submission_dir'], 'submission.zip'),
        'w',
        zipfile.ZIP_DEFLATED
    ) as zf:
        zf.write(os.path.join(config['submission_dir'], 'submission.py'), 'submission.py')
        zf.write(os.path.join(config['submission_dir'], 'weights_challenge_1.pt'), 'weights_challenge_1.pt')
        zf.write(os.path.join(config['submission_dir'], 'weights_challenge_2.pt'), 'weights_challenge_2.pt')
        
    print(f"Submission prepared at: {config['submission_dir']}/submission.zip")


def main():
    parser = argparse.ArgumentParser(description='Train EEG-ENN model for NeurIPS 2025 Challenge')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--data_release', type=str, default='R5', help='Data release version')
    parser.add_argument('--use_mini_data', action='store_true', help='Use mini dataset for testing')
    parser.add_argument('--use_mock_data', action='store_true', help='Use mock data for testing')
    parser.add_argument('--use_s3_data', action='store_true', help='Stream data from S3 (full datasets)')
    parser.add_argument('--s3_bucket', type=str, default='hbn-eeg', help='S3 bucket name')
    parser.add_argument('--max_subjects', type=int, default=None, help='Limit number of subjects (for testing)')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, default='enn', choices=['enn', 'multitask'])
    parser.add_argument('--n_channels', type=int, default=129, help='Number of EEG channels')
    parser.add_argument('--n_times', type=int, default=200, help='Number of time points')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--output_dim', type=int, default=1, help='Output dimension')
    parser.add_argument('--n_filters', type=int, default=40, help='Number of filters')
    parser.add_argument('--feature_dim', type=int, default=64, help='Feature dimension')
    
    # Subject-invariant arguments
    parser.add_argument('--use_subject_invariant', action='store_true', help='Use subject-invariant training')
    parser.add_argument('--n_subjects', type=int, default=100, help='Number of subjects')
    parser.add_argument('--lambda_reversal', type=float, default=0.1, help='Gradient reversal strength')
    parser.add_argument('--use_instance_norm', action='store_true', help='Use instance normalization')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--pretrain_epochs', type=int, default=50, help='Pretraining epochs')
    parser.add_argument('--finetune_epochs', type=int, default=30, help='Fine-tuning epochs')
    parser.add_argument('--pretrain_lr', type=float, default=1e-3, help='Pretraining learning rate')
    parser.add_argument('--finetune_lr', type=float, default=1e-4, help='Fine-tuning learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone during fine-tuning')
    parser.add_argument('--use_self_supervised', action='store_true', help='Use self-supervised pretraining')
    
    # Data split arguments
    parser.add_argument('--valid_split', type=float, default=0.15, help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.15, help='Test split ratio')
    
    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--submission_dir', type=str, default='./submission', help='Submission directory')
    parser.add_argument('--experiment_name', type=str, default='eeg_enn_experiment', help='Experiment name')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    
    args = parser.parse_args()
    config = vars(args)
    
    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['submission_dir'], exist_ok=True)
    
    # Save configuration
    with open(os.path.join(config['checkpoint_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
        
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Prepare data
    print("Preparing data...")
    loaders = prepare_data(config)
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    trainer, pretrain_history, finetune_history = train_model(model, loaders, config)
    
    # Evaluate model
    test_metrics = evaluate_model(trainer, loaders, config)
    
    # Save results (convert numpy types to native Python types for JSON serialization)
    def convert_numpy_types(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    results = {
        'config': config,
        'pretrain_history': convert_numpy_types(pretrain_history),
        'finetune_history': convert_numpy_types(finetune_history),
        'test_metrics': convert_numpy_types(test_metrics)
    }
    
    with open(os.path.join(config['checkpoint_dir'], 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
        
    # Prepare submission
    prepare_submission(model, config)
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)


if __name__ == '__main__':
    main()