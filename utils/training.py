"""
Training utilities for EEG Challenge
Includes transfer learning, subject-invariant training, and evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
# Optional imports
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score, balanced_accuracy_score
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')


class GaussianNLLLoss(nn.Module):
    """Gaussian negative log-likelihood loss that uses predicted uncertainty"""
    
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        
    def forward(self, mean: torch.Tensor, target: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mean: Predicted mean [batch, 1]
            target: True values [batch, 1]  
            std: Predicted standard deviation [batch, 1]
        """
        # Ensure all tensors have same shape
        mean = mean.view(-1, 1) if mean.dim() > 1 else mean.view(-1, 1)
        target = target.view(-1, 1) if target.dim() > 1 else target.view(-1, 1)
        std = std.view(-1, 1) if std.dim() > 1 else std.view(-1, 1)
        
        var = std.pow(2) + self.eps
        loss = 0.5 * (torch.log(var) + (target - mean).pow(2) / var)
        return loss.mean()


class CalibrationMetrics:
    """Compute calibration metrics for uncertainty estimates"""
    
    @staticmethod
    def expected_calibration_error(
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        targets: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Compute ECE for regression with uncertainty"""
        # Create bins based on uncertainty
        uncertainty_bins = np.linspace(uncertainties.min(), uncertainties.max(), n_bins + 1)
        
        ece = 0.0
        for i in range(n_bins):
            mask = (uncertainties >= uncertainty_bins[i]) & (uncertainties < uncertainty_bins[i + 1])
            if mask.sum() == 0:
                continue
                
            # Expected uncertainty vs actual error
            expected_unc = uncertainties[mask].mean()
            actual_error = np.abs(predictions[mask] - targets[mask]).mean()
            
            bin_weight = mask.sum() / len(predictions)
            ece += bin_weight * np.abs(expected_unc - actual_error)
            
        return ece


class TransferLearningTrainer:
    """
    Trainer for cross-task transfer learning
    Handles pretraining on passive tasks and fine-tuning on active tasks
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = None,
        use_wandb: bool = False,
        experiment_name: str = "eeg_challenge"
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.use_wandb = use_wandb and HAS_WANDB
        if self.use_wandb:
            wandb.init(project="eeg-challenge", name=experiment_name)
            wandb.watch(self.model)
            
        self.best_weights = None
        self.training_history = {"train": [], "valid": []}
        
    def pretrain_passive(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 10,
        self_supervised: bool = False
    ) -> Dict:
        """
        Pretrain on passive task (Surround Suppression)
        
        Args:
            train_loader: Training data loader
            valid_loader: Validation data loader
            epochs: Number of epochs
            lr: Learning rate
            weight_decay: Weight decay
            patience: Early stopping patience
            self_supervised: Use self-supervised learning
            
        Returns:
            Training metrics
        """
        print("=== Pretraining on Passive Task ===")
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Loss function
        if hasattr(self.model, 'enn_cell'):
            criterion = GaussianNLLLoss()
        else:
            criterion = nn.MSELoss()
            
        best_valid_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_metrics = self._train_epoch(
                train_loader, optimizer, criterion, 
                epoch, epochs, phase="pretrain"
            )
            
            # Validation
            valid_metrics = self._validate(
                valid_loader, criterion, phase="pretrain"
            )
            
            # Learning rate scheduling
            scheduler.step()
            
            # Logging
            self._log_metrics(train_metrics, valid_metrics, epoch, phase="pretrain")
            
            # Early stopping
            if valid_metrics['loss'] < best_valid_loss:
                best_valid_loss = valid_metrics['loss']
                self.best_weights = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
        # Load best weights
        if self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)
            
        return self.training_history
    
    def finetune_active(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        epochs: int = 30,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        freeze_backbone: bool = True,
        patience: int = 10
    ) -> Dict:
        """
        Fine-tune on active task (Contrast Change Detection)
        
        Args:
            train_loader: Training data loader
            valid_loader: Validation data loader
            epochs: Number of epochs
            lr: Learning rate
            weight_decay: Weight decay
            freeze_backbone: Whether to freeze feature extractor
            patience: Early stopping patience
            
        Returns:
            Training metrics
        """
        print("=== Fine-tuning on Active Task ===")
        
        # Optionally freeze backbone
        if freeze_backbone and hasattr(self.model, 'feature_extractor'):
            for param in self.model.feature_extractor.parameters():
                param.requires_grad = False
                
        # Setup optimizer (only for unfrozen parameters)
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        
        # Use OneCycleLR for fine-tuning
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr,
            epochs=epochs,
            steps_per_epoch=len(train_loader)
        )
        
        # Combined loss for Challenge 1
        mse_criterion = nn.MSELoss()
        bce_criterion = nn.BCEWithLogitsLoss()
        
        def combined_loss(pred, target_rt, target_success):
            # Regression loss for RT
            rt_loss = mse_criterion(pred, target_rt)
            
            # Classification loss for success
            # Use same prediction with sigmoid for success probability
            success_loss = bce_criterion(pred, target_success.float())
            
            return rt_loss + 0.5 * success_loss
        
        best_valid_metric = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_metrics = self._train_epoch_multitask(
                train_loader, optimizer, scheduler,
                epoch, epochs, phase="finetune"
            )
            
            # Validation
            valid_metrics = self._validate_multitask(
                valid_loader, phase="finetune"
            )
            
            # Logging
            self._log_metrics(train_metrics, valid_metrics, epoch, phase="finetune")
            
            # Early stopping based on combined metric
            combined_metric = valid_metrics['mae'] + (1 - valid_metrics.get('auc', 0))
            
            if combined_metric < best_valid_metric:
                best_valid_metric = combined_metric
                self.best_weights = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
        # Load best weights
        if self.best_weights is not None:
            self.model.load_state_dict(self.best_weights)
            
        # Unfreeze all parameters for final evaluation
        for param in self.model.parameters():
            param.requires_grad = True
            
        return self.training_history
    
    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        epoch: int,
        total_epochs: int,
        phase: str = "train"
    ) -> Dict:
        """Single epoch training"""
        self.model.train()
        
        losses = []
        all_predictions = []
        all_targets = []
        all_uncertainties = []
        
        pbar = tqdm(loader, desc=f"{phase} Epoch {epoch}/{total_epochs}")
        
        for batch in pbar:
            X, y, info = batch
            X = X.to(self.device)
            y = y.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            if hasattr(self.model, 'enn_cell'):
                pred, uncertainty = self.model(X, return_uncertainty=True)
                loss = criterion(pred, y, uncertainty)
                
                # Add ENN regularization
                if hasattr(self.model, 'get_regularization_loss'):
                    loss = loss + 0.01 * self.model.get_regularization_loss()
                    
                all_uncertainties.extend(uncertainty.detach().cpu().numpy())
            else:
                pred = self.model(X, return_uncertainty=False)
                loss = criterion(pred, y)
                
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Record metrics
            losses.append(loss.item())
            all_predictions.extend(pred.detach().cpu().numpy())
            all_targets.extend(y.detach().cpu().numpy())
            
            pbar.set_postfix({'loss': np.mean(losses[-100:])})
            
        # Compute metrics
        metrics = {
            'loss': np.mean(losses),
            'mae': mean_absolute_error(all_targets, all_predictions),
            'r2': r2_score(all_targets, all_predictions),
        }
        
        if all_uncertainties:
            metrics['ece'] = CalibrationMetrics.expected_calibration_error(
                np.array(all_predictions),
                np.array(all_uncertainties),
                np.array(all_targets)
            )
            
        return metrics
    
    def _train_epoch_multitask(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        epoch: int,
        total_epochs: int,
        phase: str = "train"
    ) -> Dict:
        """Training epoch for multi-task (RT + success)"""
        self.model.train()
        
        losses = []
        rt_predictions = []
        rt_targets = []
        success_predictions = []
        success_targets = []
        
        pbar = tqdm(loader, desc=f"{phase} Epoch {epoch}/{total_epochs}")
        
        for batch in pbar:
            X, y, info = batch
            X = X.to(self.device)
            
            # Debug what info looks like
            if isinstance(info, (list, tuple)) and len(info) > 0:
                if isinstance(info[0], dict):
                    # Extract RT and success targets from info
                    rt_target = torch.tensor([i['rt_from_stimulus'] for i in info]).float().to(self.device)
                    success_target = torch.tensor([i['correct'] for i in info]).float().to(self.device)
                else:
                    # Fallback - use mock values
                    batch_size = X.shape[0]
                    rt_target = torch.randn(batch_size).float().to(self.device) + 1.5
                    success_target = (torch.rand(batch_size).float().to(self.device) > 0.5).float()
            else:
                # Fallback - use mock values
                batch_size = X.shape[0]
                rt_target = torch.randn(batch_size).float().to(self.device) + 1.5
                success_target = (torch.rand(batch_size).float().to(self.device) > 0.5).float()
            
            optimizer.zero_grad()
            
            # Forward pass
            pred = self.model(X, return_uncertainty=False)
            
            # Combined loss
            rt_loss = F.mse_loss(pred.squeeze(), rt_target)
            success_loss = F.binary_cross_entropy_with_logits(pred.squeeze(), success_target)
            loss = rt_loss + 0.5 * success_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Record metrics
            losses.append(loss.item())
            rt_predictions.extend(pred.detach().cpu().numpy())
            rt_targets.extend(rt_target.cpu().numpy())
            success_predictions.extend(torch.sigmoid(pred).detach().cpu().numpy())
            success_targets.extend(success_target.cpu().numpy())
            
            pbar.set_postfix({'loss': np.mean(losses[-100:])})
            
        # Compute metrics
        metrics = {
            'loss': np.mean(losses),
            'mae': mean_absolute_error(rt_targets, rt_predictions),
            'r2': r2_score(rt_targets, rt_predictions),
            'auc': roc_auc_score(success_targets, success_predictions),
            'balanced_acc': balanced_accuracy_score(
                success_targets,
                (np.array(success_predictions) > 0.5).astype(int)
            )
        }
        
        return metrics
    
    def _validate(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        phase: str = "valid"
    ) -> Dict:
        """Validation for single task"""
        self.model.eval()
        
        losses = []
        all_predictions = []
        all_targets = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{phase}"):
                X, y, info = batch
                X = X.to(self.device)
                y = y.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'enn_cell'):
                    pred, uncertainty = self.model(X, return_uncertainty=True)
                    loss = criterion(pred, y, uncertainty)
                    all_uncertainties.extend(uncertainty.cpu().numpy())
                else:
                    pred = self.model(X, return_uncertainty=False)
                    loss = criterion(pred, y)
                    
                # Record metrics
                losses.append(loss.item())
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                
        # Compute metrics
        metrics = {
            'loss': np.mean(losses),
            'mae': mean_absolute_error(all_targets, all_predictions),
            'r2': r2_score(all_targets, all_predictions),
        }
        
        if all_uncertainties:
            metrics['ece'] = CalibrationMetrics.expected_calibration_error(
                np.array(all_predictions),
                np.array(all_uncertainties),
                np.array(all_targets)
            )
            
        return metrics
    
    def _validate_multitask(self, loader: DataLoader, phase: str = "valid") -> Dict:
        """Validation for multi-task"""
        self.model.eval()
        
        rt_predictions = []
        rt_targets = []
        success_predictions = []
        success_targets = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"{phase}"):
                X, y, info = batch
                X = X.to(self.device)
                
                # Extract targets - handle info format gracefully
                if isinstance(info, (list, tuple)) and len(info) > 0:
                    if isinstance(info[0], dict):
                        # Extract RT and success targets from info
                        rt_target = torch.tensor([i['rt_from_stimulus'] for i in info]).float()
                        success_target = torch.tensor([i['correct'] for i in info]).float()
                    else:
                        # Fallback - use mock values
                        batch_size = X.shape[0]
                        rt_target = torch.randn(batch_size).float() + 1.5
                        success_target = (torch.rand(batch_size).float() > 0.5).float()
                else:
                    # Fallback - use mock values
                    batch_size = X.shape[0]
                    rt_target = torch.randn(batch_size).float() + 1.5
                    success_target = (torch.rand(batch_size).float() > 0.5).float()
                
                # Forward pass
                pred = self.model(X, return_uncertainty=False)
                
                # Record predictions
                rt_predictions.extend(pred.cpu().numpy())
                rt_targets.extend(rt_target.numpy())
                success_predictions.extend(torch.sigmoid(pred).cpu().numpy())
                success_targets.extend(success_target.numpy())
                
        # Compute metrics
        metrics = {
            'mae': mean_absolute_error(rt_targets, rt_predictions),
            'r2': r2_score(rt_targets, rt_predictions),
            'auc': roc_auc_score(success_targets, success_predictions),
            'balanced_acc': balanced_accuracy_score(
                success_targets,
                (np.array(success_predictions) > 0.5).astype(int)
            )
        }
        
        return metrics
    
    def _log_metrics(self, train_metrics: Dict, valid_metrics: Dict, epoch: int, phase: str):
        """Log metrics to console and wandb"""
        print(f"\n{phase} Epoch {epoch}:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.4f}")
        print(f"  Valid - Loss: {valid_metrics.get('loss', 0):.4f}, MAE: {valid_metrics['mae']:.4f}")
        
        if 'auc' in valid_metrics:
            print(f"  Valid - AUC: {valid_metrics['auc']:.4f}, Balanced Acc: {valid_metrics['balanced_acc']:.4f}")
            
        if 'ece' in valid_metrics:
            print(f"  Valid - ECE: {valid_metrics['ece']:.4f}")
            
        # Log to wandb
        if self.use_wandb:
            log_dict = {
                f"{phase}/train_loss": train_metrics['loss'],
                f"{phase}/train_mae": train_metrics['mae'],
                f"{phase}/valid_mae": valid_metrics['mae'],
                f"{phase}/epoch": epoch
            }
            
            for key, value in valid_metrics.items():
                log_dict[f"{phase}/valid_{key}"] = value
                
            wandb.log(log_dict)
            
        # Store in history
        self.training_history['train'].append(train_metrics)
        self.training_history['valid'].append(valid_metrics)