"""
BEF Training Pipeline with Pretrain/Fine-tune Strategy
Implements staged training for optimal EEG performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List
import yaml

# Optional wandb import
try:
    import wandb
except ImportError:
    wandb = None

try:
    from .pipeline import BEF_EEG, MultiTaskBEF, PretrainableBEF
except ImportError:
    from pipeline import BEF_EEG, MultiTaskBEF, PretrainableBEF

try:
    from .utils_io import load_eeg_data
except ImportError:
    from utils_io import load_eeg_data



class BEFTrainer:
    """
    Trainer for BEF pipeline with staged training strategy
    """
    
    def __init__(
        self,
        model: BEF_EEG,
        config: Dict,
        device: str = "cuda",  # Force GPU usage - we have 3 GPUs available
        use_wandb: bool = True
    ):
        # Device diagnostics
        print(f"=== Device Setup ===")
        print(f"CUDA available: {torch.cuda.is_available()}")

        # Force GPU usage - we have 3 GPUs (nvidia0, nvidia2, nvidia3)
        try:
            if device.startswith("cuda"):
                gpu_id = 0 if device == "cuda" else int(device.split(":")[1])
                device_obj = torch.device(f"cuda:{gpu_id}")
                print(f"Forcing GPU usage: {device_obj}")
                # Test GPU with a small tensor
                test_tensor = torch.randn(10, device=device_obj)
                print(f"✅ GPU {gpu_id} is working: tensor on {test_tensor.device}")
                device = str(device_obj)
            else:
                device_obj = torch.device(device)
        except Exception as e:
            print(f"❌ GPU {device} failed: {e}")
            print("Falling back to CPU")
            device = "cpu"
            device_obj = torch.device("cpu")

        print(f"Final device: {device}")

        self.model = model.to(device_obj)
        self.config = config
        self.task = config.get("task", "regression").lower()  # Explicit task specification
        self.num_classes = int(config.get("num_classes", 1))  # Number of classes for multiclass
        self.device = device_obj
        self.use_wandb = use_wandb
        
        if use_wandb and wandb is not None:
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
        
    def setup_optimizers(self, stage: str = "pretrain", head_only: bool = False):
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
            # Lower LR for stability
            self.optimizers['fusion'] = optim.AdamW(
                self.model.fusion.parameters(),
                lr=self.config.get('lr_fusion', 1e-4),  # Reduced from 5e-4
                weight_decay=self.config.get('weight_decay', 1e-4)
            )
            
        elif stage == "finetune_full":
            # Fine-tune stage 2: Train full pipeline
            # Even lower LR for final fine-tuning
            self.optimizers['full'] = optim.AdamW(
                self.model.parameters(),
                lr=self.config.get('lr_finetune', 3e-5),  # Reduced from 1e-4
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
            
            if self.use_wandb and wandb is not None and batch_idx % 10 == 0:
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
        optimizer = self.optimizers.get(stage, self.optimizers.get('full', list(self.optimizers.values())[0]))
        
        losses = {
            'total': [],
            'mse': [],
            'nll': [],
            'reg': []
        }
        
        for batch_idx, (data, target) in enumerate(tqdm(loader, desc=f"Train Epoch {epoch}")):
            data = data.to(self.device).float()
            target = target.to(self.device).float()
            
            # Forward pass - use mc_samples=1 for training (faster gradients)
            outputs = self.model(data, mc_samples=1)
            
            # Compute losses - handle DataParallel
            model_for_loss = self.model.module if hasattr(self.model, 'module') else self.model
            loss_dict = model_for_loss.compute_loss(
                outputs, target,
                task=("classification" if self.task == "binary"
                      else "multiclass" if self.task == "multiclass"
                      else "regression")
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
            
            if self.use_wandb and wandb is not None and batch_idx % 10 == 0:
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

            # Get predictions with uncertainty - handle DataParallel
            model_for_pred = self.model.module if hasattr(self.model, 'module') else self.model
            pred, aleatoric, epistemic = model_for_pred.predict_with_uncertainty(data, n_samples=100)
            uncertainty = aleatoric + epistemic  # Total uncertainty

            predictions.append(pred.cpu())
            targets.append(target.cpu())
            uncertainties.append(uncertainty.cpu())

        predictions = torch.cat(predictions)
        targets = torch.cat(targets)
        uncertainties = torch.cat(uncertainties)

        # Sanity check: print target statistics to verify task type
        print(f"[{stage}] target stats: min={targets.min().item():.3f}, max={targets.max().item():.3f}, uniq≈{targets.unique().numel()}")

        # Additional diagnostics for multiclass
        if self.task == "multiclass":
            target_counts = torch.bincount(targets.long().view(-1))
            print(f"[{stage}] class distribution: {target_counts.tolist()}")

        # Compute metrics - pass predictions_are_logits=True for classification tasks
        metrics = self.compute_metrics(predictions, targets, uncertainties,
                                        predictions_are_logits=(self.task in ["binary", "multiclass"]))
        
        if self.use_wandb and wandb is not None:
            wandb.log({f'{stage}_{k}': v for k, v in metrics.items()})
        
        return metrics
    
    def compute_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: torch.Tensor,
        predictions_are_logits: bool = False
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics - Fixed to use task from config
        """
        # Ensure tensors are squeezed and on CPU
        predictions = predictions.squeeze().cpu()
        targets = targets.squeeze().cpu()
        uncertainties = uncertainties.squeeze().cpu()

        metrics = {}

        # Use explicit task from config instead of guessing
        is_binary = (self.task == "binary")
        is_multiclass = (self.task == "multiclass")

        if is_multiclass:
            # MULTICLASS METRICS
            if predictions.dim() == 1:  # If predictions are 1D, they might be class indices
                preds = predictions.long()
                logits = None
                probs = None
            elif predictions.dim() == 2 and predictions.shape[-1] == self.num_classes:
                # predictions are logits -> softmax
                logits = predictions.float()
                probs = torch.softmax(logits, dim=-1)
                preds = probs.argmax(dim=-1)
            else:
                # Handle unexpected shape
                print(f"Warning: unexpected prediction shape {predictions.shape} for multiclass")
                preds = predictions.squeeze().long()
                probs = None

            targets_long = targets.long().view(-1)
            acc = (preds == targets_long).float().mean().item()

            # Cross-entropy loss if we have logits
            if logits is not None and predictions_are_logits:
                ce_loss = F.cross_entropy(logits, targets_long).item()
            else:
                ce_loss = float('nan')

            # Macro F1 score
            try:
                from sklearn.metrics import f1_score
                f1 = f1_score(targets_long.numpy(), preds.numpy(), average="macro", zero_division=0)
            except Exception as e:
                print(f"F1 calculation failed: {e}")
                f1 = float("nan")

            metrics.update({
                "acc": acc,
                "f1_macro": f1,
                "ce_loss": ce_loss,
                "mean_prob_max": probs.max(dim=-1).values.mean().item() if probs is not None else float('nan')
            })

        elif is_binary:
            # CLASSIFICATION METRICS
            if predictions_are_logits:
                logits = predictions.float()
                probs = torch.sigmoid(logits)
            else:
                probs = torch.clamp(predictions.float(), 1e-7, 1 - 1e-7)
                logits = None

            # Ensure clean binary targets
            targets_binary = (targets > 0.5).float()
            mae = torch.mean(torch.abs(probs - targets_binary)).item()  # on [0,1]

            # BCE with safe computation
            if predictions_are_logits:
                bce = F.binary_cross_entropy_with_logits(
                    logits.squeeze(), targets_binary.squeeze()
                ).item()
            else:
                bce = F.binary_cross_entropy(
                    probs.squeeze(), targets_binary.squeeze()
                ).item()

            # AUC (if we have both classes)
            try:
                from sklearn.metrics import roc_auc_score
                import numpy as np
                targets_np = targets_binary.numpy().ravel()
                probs_np = probs.numpy().ravel()
                # Check if we have both classes
                if len(np.unique(targets_np)) > 1:
                    auc = roc_auc_score(targets_np, probs_np)
                else:
                    auc = float("nan")  # single-class val fold
            except Exception as e:
                print(f"AUC calculation failed: {e}")
                auc = 0.0

            metrics.update({
                'mae': mae,  # MAE on probabilities (should be ≤ 1.0)
                'bce': bce,  # Binary cross entropy
                'auc': auc,
                'mean_prob': probs.mean().item(),
                'prob_std': probs.std().item()
            })

        else:
            # REGRESSION METRICS
            mae = torch.mean(torch.abs(predictions - targets)).item()
            mse = torch.mean((predictions - targets) ** 2).item()

            # R-squared (handle edge cases)
            ss_tot = torch.sum((targets - targets.mean()) ** 2)
            if ss_tot > 1e-8:
                ss_res = torch.sum((targets - predictions) ** 2)
                r2 = 1 - (ss_res / ss_tot).item()
            else:
                r2 = 0.0

            metrics.update({
                'mae': mae,
                'mse': mse,
                'r2': r2
            })

        # Uncertainty metrics (keep simple/finite)
        try:
            if torch.isfinite(uncertainties).all() and uncertainties.numel() > 0:
                metrics['mean_uncertainty'] = uncertainties.mean().item()
                metrics['uncertainty_std'] = (uncertainties.std().item() if uncertainties.numel() > 1 else 0.0)
            else:
                metrics['mean_uncertainty'] = float('nan')
                metrics['uncertainty_std'] = float('nan')
        except:
            metrics['mean_uncertainty'] = float('nan')
            metrics['uncertainty_std'] = float('nan')

        # Optional toy ECE only for binary; otherwise NaN
        try:
            if is_binary:
                confidence = torch.abs(probs - 0.5) * 2
                correct = (probs > 0.5) == (targets_binary > 0.5)
                accuracy = correct.float()
                ece = torch.mean(torch.abs(confidence - accuracy)).item() if confidence.numel() > 0 else float('nan')
            else:
                ece = float('nan')
        except:
            ece = float('nan')

        metrics['ece'] = ece

        return metrics
    
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

        best_val_metric = float('inf')
        patience_counter = 0
        patience = 5  # Early stopping patience

        for epoch in range(self.config.get('epochs_fusion', 20)):
            train_metrics = self.train_epoch(train_loader, epoch, 'fusion')
            val_metrics = self.evaluate(val_loader, 'fusion_val')

            # Use proper metric for monitoring
            if 'auc' in val_metrics and val_metrics['auc'] > 0:
                # Classification: monitor AUC (higher is better)
                val_monitor = -val_metrics['auc']  # Negative for minimization
                print(f"Epoch {epoch}: Train Loss {train_metrics['total']:.4f}, Val AUC {val_metrics['auc']:.4f}, Val MAE(probs) {val_metrics['mae']:.4f}")
            else:
                # Regression: monitor MAE
                val_monitor = val_metrics['mae']
                print(f"Epoch {epoch}: Train Loss {train_metrics['total']:.4f}, Val MAE {val_metrics['mae']:.4f}")

            # Early stopping and best model saving
            if val_monitor < best_val_metric:
                best_val_metric = val_monitor
                patience_counter = 0
                self.best_metrics['val_mae'] = val_metrics['mae']
                self.save_checkpoint('best_fusion.pt')
                print(f"  -> New best model saved!")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  -> Early stopping after {patience} epochs without improvement")
                    break
        
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
        
        if self.use_wandb and wandb is not None:
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
    import os
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
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
        config,
        batch_size=config['train']['batch_size'],
        num_workers=config['train'].get('num_workers', 4)
    )
    
    # Create trainer
    trainer = BEFTrainer(model, config, use_wandb=config.get('use_wandb', True))
    
    # Train model
    best_metrics = trainer.train(train_loader, val_loader, test_loader)
    
    print("\n=== Training Complete ===")
    print(f"Best Metrics: {best_metrics}")


if __name__ == "__main__":
    main()
