"""
BEF Training Pipeline with Pretrain/Fine-tune Strategy
Implements staged training for optimal EEG performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import wandb
from typing import Dict, Optional, Tuple, List
import yaml

from pipeline import BEF_EEG, MultiTaskBEF, PretrainableBEF
from utils_io import load_eeg_data


class BEFTrainer:
    """
    Trainer for BEF pipeline with staged training strategy
    """
    
    def __init__(
        self,
        model: BEF_EEG,
        config: Dict,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_wandb: bool = True
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        if use_wandb:
            wandb.init(project="eeg-bef", config=config)
            wandb.watch(model)
        
        # Optimizers for different stages
        self.optimizers = {}
        self.schedulers = {}
        
        # Metrics tracking
        self.best_metrics = {
            'val_mae': float('inf'),
            'val_r2': -float('inf'),
            'val_auc': 0.0
        }
        
    def setup_optimizers(self, stage: str = "pretrain"):
        """Setup optimizers for different training stages"""
        
        if stage == "pretrain":
            # Pretrain: Focus on ENN representation learning
            params = list(self.model.enn.parameters())
            if hasattr(self.model, 'contrastive_head'):
                params += list(self.model.contrastive_head.parameters())
            
            self.optimizers['pretrain'] = optim.AdamW(
                params,
                lr=self.config.get('lr_pretrain', 1e-3),
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
            
        elif stage == "finetune_fusion":
            # Fine-tune stage 1: Train Fusion Alpha with frozen ENN
            self.optimizers['fusion'] = optim.AdamW(
                self.model.fusion.parameters(),
                lr=self.config.get('lr_fusion', 5e-4),
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
            
        elif stage == "finetune_full":
            # Fine-tune stage 2: Train full pipeline
            self.optimizers['full'] = optim.AdamW(
                self.model.parameters(),
                lr=self.config.get('lr_finetune', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
            
            # Cosine annealing scheduler
            self.schedulers['full'] = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizers['full'],
                T_max=self.config.get('epochs_finetune', 50)
            )
    
    def freeze_components(self, components: List[str]):
        """Freeze specific model components"""
        
        for comp in components:
            if comp == "bicep" and hasattr(self.model, 'bicep'):
                for param in self.model.bicep.parameters():
                    param.requires_grad = False
                    
            elif comp == "enn" and hasattr(self.model, 'enn'):
                for param in self.model.enn.parameters():
                    param.requires_grad = False
                    
            elif comp == "fusion" and hasattr(self.model, 'fusion'):
                for param in self.model.fusion.parameters():
                    param.requires_grad = False
    
    def unfreeze_components(self, components: List[str]):
        """Unfreeze specific model components"""
        
        for comp in components:
            if comp == "bicep" and hasattr(self.model, 'bicep'):
                for param in self.model.bicep.parameters():
                    param.requires_grad = True
                    
            elif comp == "enn" and hasattr(self.model, 'enn'):
                for param in self.model.enn.parameters():
                    param.requires_grad = True
                    
            elif comp == "fusion" and hasattr(self.model, 'fusion'):
                for param in self.model.fusion.parameters():
                    param.requires_grad = True
    
    def pretrain_epoch(
        self,
        loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Pretrain epoch using contrastive learning
        """
        self.model.train()
        optimizer = self.optimizers['pretrain']
        
        losses = []
        
        for batch_idx, (data, _) in enumerate(tqdm(loader, desc=f"Pretrain Epoch {epoch}")):
            data = data.to(self.device).float()
            
            # Forward pass with augmentation
            if isinstance(self.model, PretrainableBEF):
                outputs = self.model.pretrain_forward(data, augment=True)
                loss = outputs['contrastive_loss']
            else:
                # Standard pretraining: predict noise level
                noise = torch.randn_like(data) * 0.1
                noisy_data = data + noise
                
                outputs = self.model(noisy_data)
                # Predict noise magnitude
                target = noise.std(dim=[1, 2])
                pred = outputs['total_uncertainty'].squeeze()
                loss = nn.functional.mse_loss(pred, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            losses.append(loss.item())
            
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'pretrain_loss': loss.item(),
                    'pretrain_epoch': epoch
                })
        
        return {'pretrain_loss': np.mean(losses)}
    
    def train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        stage: str = "finetune_full"
    ) -> Dict[str, float]:
        """
        Standard training epoch
        """
        self.model.train()
        optimizer = self.optimizers.get(stage, self.optimizers['full'])
        
        losses = {
            'total': [],
            'mse': [],
            'nll': [],
            'reg': []
        }
        
        for batch_idx, (data, target) in enumerate(tqdm(loader, desc=f"Train Epoch {epoch}")):
            data = data.to(self.device).float()
            target = target.to(self.device).float()
            
            # Forward pass
            outputs = self.model(data, mc_samples=4)
            
            # Compute losses
            loss_dict = self.model.compute_loss(
                outputs, target,
                task="regression" if target.dim() == 1 else "classification"
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            # Track losses
            for k, v in loss_dict.items():
                if k in losses:
                    losses[k].append(v.item())
            
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    f'{stage}_loss': loss_dict['total'].item(),
                    f'{stage}_epoch': epoch
                })
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in losses.items() if v}
        
        return avg_losses
    
    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        stage: str = "val"
    ) -> Dict[str, float]:
        """
        Evaluate model performance
        """
        self.model.eval()
        
        predictions = []
        targets = []
        uncertainties = []
        
        for data, target in tqdm(loader, desc=f"Evaluating {stage}"):
            data = data.to(self.device).float()
            target = target.to(self.device).float()
            
            # Get predictions with uncertainty
            pred, uncertainty = self.model.predict_with_uncertainty(data, n_samples=100)
            
            predictions.append(pred.cpu())
            targets.append(target.cpu())
            uncertainties.append(uncertainty.cpu())
        
        predictions = torch.cat(predictions)
        targets = torch.cat(targets)
        uncertainties = torch.cat(uncertainties)
        
        # Compute metrics
        metrics = self.compute_metrics(predictions, targets, uncertainties)
        
        if self.use_wandb:
            wandb.log({f'{stage}_{k}': v for k, v in metrics.items()})
        
        return metrics
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics
        """
        # Regression metrics
        mae = torch.mean(torch.abs(predictions - targets)).item()
        mse = torch.mean((predictions - targets) ** 2).item()
        
        # R-squared
        ss_res = torch.sum((targets - predictions) ** 2)
        ss_tot = torch.sum((targets - targets.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot).item()
        
        # Calibration metrics
        # Compute normalized calibration error
        sorted_idx = torch.argsort(uncertainties.squeeze())
        sorted_targets = targets[sorted_idx]
        sorted_preds = predictions[sorted_idx]
        sorted_unc = uncertainties[sorted_idx]
        
        # Bin predictions by uncertainty
        n_bins = 10
        bin_size = len(sorted_targets) // n_bins
        calibration_errors = []
        
        for i in range(n_bins):
            start = i * bin_size
            end = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_targets)
            
            bin_preds = sorted_preds[start:end]
            bin_targets = sorted_targets[start:end]
            bin_unc = sorted_unc[start:end]
            
            # Expected uncertainty should match actual error
            expected_error = bin_unc.mean()
            actual_error = torch.abs(bin_preds - bin_targets).mean()
            calibration_errors.append(torch.abs(expected_error - actual_error))
        
        ece = torch.mean(torch.stack(calibration_errors)).item()
        
        # AUC for binary classification (if applicable)
        if targets.unique().numel() == 2:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(targets.numpy(), torch.sigmoid(predictions).numpy())
        else:
            auc = 0.0
        
        return {
            'mae': mae,
            'mse': mse,
            'r2': r2,
            'ece': ece,
            'auc': auc,
            'mean_uncertainty': uncertainties.mean().item()
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None
    ):
        """
        Full training pipeline with pretrain and fine-tune
        """
        
        # Stage 1: Pretrain (if configured)
        if self.config.get('do_pretrain', True):
            print("\n=== Stage 1: Pretraining ===")
            self.setup_optimizers('pretrain')
            
            for epoch in range(self.config.get('epochs_pretrain', 10)):
                train_metrics = self.pretrain_epoch(train_loader, epoch)
                print(f"Epoch {epoch}: {train_metrics}")
                
                if epoch % 5 == 0:
                    val_metrics = self.evaluate(val_loader, 'pretrain_val')
                    print(f"Validation: {val_metrics}")
        
        # Stage 2: Fine-tune Fusion Alpha
        print("\n=== Stage 2: Fine-tuning Fusion Alpha ===")
        self.setup_optimizers('finetune_fusion')
        self.freeze_components(['enn', 'bicep'])
        
        for epoch in range(self.config.get('epochs_fusion', 20)):
            train_metrics = self.train_epoch(train_loader, epoch, 'fusion')
            val_metrics = self.evaluate(val_loader, 'fusion_val')
            
            print(f"Epoch {epoch}: Train {train_metrics['total']:.4f}, Val MAE {val_metrics['mae']:.4f}")
            
            # Save best model
            if val_metrics['mae'] < self.best_metrics['val_mae']:
                self.best_metrics['val_mae'] = val_metrics['mae']
                self.save_checkpoint('best_fusion.pt')
        
        # Stage 3: Fine-tune full model
        print("\n=== Stage 3: Fine-tuning Full Model ===")
        self.setup_optimizers('finetune_full')
        self.unfreeze_components(['enn'])  # Keep BICEP frozen
        
        for epoch in range(self.config.get('epochs_finetune', 50)):
            train_metrics = self.train_epoch(train_loader, epoch, 'full')
            val_metrics = self.evaluate(val_loader, 'val')
            
            # Update learning rate
            if 'full' in self.schedulers:
                self.schedulers['full'].step()
            
            print(f"Epoch {epoch}: Train {train_metrics['total']:.4f}, "
                  f"Val MAE {val_metrics['mae']:.4f}, R2 {val_metrics['r2']:.4f}")
            
            # Save best model
            if val_metrics['mae'] < self.best_metrics['val_mae']:
                self.best_metrics.update(val_metrics)
                self.save_checkpoint('best_full.pt')
        
        # Final evaluation on test set
        if test_loader is not None:
            print("\n=== Final Test Evaluation ===")
            self.load_checkpoint('best_full.pt')
            test_metrics = self.evaluate(test_loader, 'test')
            print(f"Test Results: {test_metrics}")
            
            if self.use_wandb:
                wandb.summary.update({f'test_{k}': v for k, v in test_metrics.items()})
        
        return self.best_metrics
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_metrics': self.best_metrics
        }
        
        path = Path('checkpoints') / filename
        path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, path)
        
        if self.use_wandb:
            wandb.save(str(path))
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = Path('checkpoints') / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_metrics = checkpoint.get('best_metrics', self.best_metrics)


def main():
    """Main training script"""
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = PretrainableBEF(
        in_chans=config['in_chans'],
        sfreq=config['sfreq'],
        n_paths=config['bicep']['n_paths'],
        K=config['enn']['K'],
        gnn_hidden=config['fusion']['gnn_hid'],
        gnn_layers=config['fusion']['layers'],
        dropout=config['fusion']['dropout']
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load data
    train_loader, val_loader, test_loader = load_eeg_data(
        config['data_path'],
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 4)
    )
    
    # Create trainer
    trainer = BEFTrainer(model, config, use_wandb=config.get('use_wandb', True))
    
    # Train model
    best_metrics = trainer.train(train_loader, val_loader, test_loader)
    
    print("\n=== Training Complete ===")
    print(f"Best Metrics: {best_metrics}")


if __name__ == "__main__":
    main()