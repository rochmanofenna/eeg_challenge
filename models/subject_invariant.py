"""
Subject-invariant training components for EEG Challenge
Includes gradient reversal, domain adaptation, and mixup strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import numpy as np


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal layer for domain adaptation"""
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Wrapper for gradient reversal"""
    
    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_)


class SubjectDiscriminator(nn.Module):
    """
    Subject classifier to be confused by gradient reversal
    """
    
    def __init__(self, input_dim: int, n_subjects: int, hidden_dim: int = 128):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, n_subjects)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SubjectInvariantEEGENN(nn.Module):
    """
    Subject-invariant version of EEG-ENN model
    Uses gradient reversal to learn features that can't predict subject
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        n_subjects: int,
        feature_dim: int = 64,
        lambda_reversal: float = 0.1
    ):
        super().__init__()
        
        self.base_model = base_model
        self.n_subjects = n_subjects
        
        # Gradient reversal layer
        self.grl = GradientReversalLayer(lambda_reversal)
        
        # Subject discriminator
        self.subject_discriminator = SubjectDiscriminator(feature_dim, n_subjects)
        
        # Store lambda for scheduling
        self.lambda_reversal = lambda_reversal
        
    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False,
        return_uncertainty: bool = True
    ) -> torch.Tensor | Dict[str, torch.Tensor]:
        """
        Forward pass with subject-invariant training
        
        Args:
            x: EEG data [batch, channels, time]
            return_all: Whether to return all outputs for training
            
        Returns:
            Task predictions or dict with all outputs
        """
        # Get features from base model
        if hasattr(self.base_model, 'feature_extractor'):
            features = self.base_model.feature_extractor(x)
            features_proj = self.base_model.projection(features)
            
            # Task prediction
            if hasattr(self.base_model, 'enn_cell'):
                task_pred, task_uncertainty = self.base_model.enn_cell(features_proj)
            else:
                task_pred = self.base_model.enn_cell(features_proj)
                task_uncertainty = None
        else:
            # Fallback for models without explicit feature extraction
            task_pred = self.base_model(x, return_uncertainty=return_uncertainty)
            features_proj = task_pred  # Use predictions as features
            task_uncertainty = None
            
        if not return_all:
            return task_pred
            
        # Subject prediction (with reversed gradients)
        features_reversed = self.grl(features_proj)
        subject_logits = self.subject_discriminator(features_reversed)
        
        outputs = {
            'task_pred': task_pred,
            'subject_logits': subject_logits,
            'features': features_proj
        }
        
        if task_uncertainty is not None:
            outputs['task_uncertainty'] = task_uncertainty
            
        return outputs
    
    def update_lambda(self, epoch: int, max_epochs: int):
        """
        Update gradient reversal strength during training
        Uses schedule from DANN paper
        """
        p = float(epoch) / max_epochs
        lambda_new = 2. / (1. + np.exp(-10. * p)) - 1
        self.grl.lambda_ = lambda_new * self.lambda_reversal


class MixUpAugmentation:
    """
    MixUp augmentation for better generalization across subjects
    """
    
    def __init__(self, alpha: float = 0.2, cross_subject: bool = True):
        self.alpha = alpha
        self.cross_subject = cross_subject
        
    def __call__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        subject_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply mixup augmentation
        
        Args:
            x: Input data [batch, ...]
            y: Labels [batch, ...]
            subject_ids: Subject IDs for cross-subject mixing
            
        Returns:
            mixed_x, mixed_y, lambda
        """
        batch_size = x.size(0)
        
        # Sample lambda from Beta distribution
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1
            
        # Get permutation indices
        if self.cross_subject and subject_ids is not None:
            # Ensure we mix across different subjects
            index = self._get_cross_subject_permutation(subject_ids)
        else:
            index = torch.randperm(batch_size).to(x.device)
            
        # Mix inputs and targets
        mixed_x = lam * x + (1 - lam) * x[index]
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_x, mixed_y, torch.tensor(lam)
    
    def _get_cross_subject_permutation(self, subject_ids: torch.Tensor) -> torch.Tensor:
        """Get permutation that ensures different subjects are mixed"""
        batch_size = len(subject_ids)
        index = torch.arange(batch_size)
        
        # Shuffle until we have all different subjects
        for i in range(batch_size):
            candidates = torch.where(subject_ids != subject_ids[i])[0]
            if len(candidates) > 0:
                j = candidates[torch.randint(len(candidates), (1,))].item()
                index[i] = j
                
        return index


class InstanceNormPerChannel(nn.Module):
    """
    Instance normalization per EEG channel
    Helps with inter-subject amplitude differences
    """
    
    def __init__(self, n_channels: int, eps: float = 1e-5):
        super().__init__()
        self.n_channels = n_channels
        self.eps = eps
        
        # Learnable affine parameters per channel
        self.weight = nn.Parameter(torch.ones(n_channels, 1))
        self.bias = nn.Parameter(torch.zeros(n_channels, 1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply instance norm per channel
        x: [batch, channels, time]
        """
        # Compute stats per channel
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply affine transform
        x_norm = x_norm * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        
        return x_norm


class SubjectAdaptiveBatchNorm(nn.Module):
    """
    Adaptive batch normalization that can adjust to new subjects
    """
    
    def __init__(self, num_features: int, momentum: float = 0.1):
        super().__init__()
        
        # Standard batch norm
        self.bn = nn.BatchNorm1d(num_features, momentum=momentum)
        
        # Subject-specific scaling
        self.subject_scale = nn.Parameter(torch.ones(1, num_features))
        self.subject_shift = nn.Parameter(torch.zeros(1, num_features))
        
        # For test-time adaptation
        self.adaptation_rate = 0.1
        
    def forward(self, x: torch.Tensor, adapt: bool = False) -> torch.Tensor:
        """
        Forward with optional test-time adaptation
        """
        # Standard batch norm
        x_norm = self.bn(x)
        
        if adapt and not self.training:
            # Test-time adaptation: slowly adjust to new subject statistics
            with torch.no_grad():
                batch_mean = x.mean(dim=0)
                batch_var = x.var(dim=0, unbiased=False)
                
                # Update running stats
                self.bn.running_mean = (
                    (1 - self.adaptation_rate) * self.bn.running_mean +
                    self.adaptation_rate * batch_mean
                )
                self.bn.running_var = (
                    (1 - self.adaptation_rate) * self.bn.running_var +
                    self.adaptation_rate * batch_var
                )
                
        # Apply subject-specific transform
        x_norm = x_norm * self.subject_scale + self.subject_shift
        
        return x_norm


class SubjectInvariantLoss(nn.Module):
    """
    Combined loss for subject-invariant training
    """
    
    def __init__(
        self,
        task_weight: float = 1.0,
        subject_weight: float = 0.1,
        consistency_weight: float = 0.1
    ):
        super().__init__()
        
        self.task_weight = task_weight
        self.subject_weight = subject_weight
        self.consistency_weight = consistency_weight
        
        # Task losses
        self.task_criterion = nn.MSELoss()
        self.subject_criterion = nn.CrossEntropyLoss()
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mixed_outputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss
        
        Args:
            outputs: Model outputs dict
            targets: Target values dict
            mixed_outputs: Outputs from mixup samples
            
        Returns:
            Dict with individual losses and total loss
        """
        losses = {}
        
        # Task loss
        task_loss = self.task_criterion(outputs['task_pred'], targets['task'])
        losses['task'] = task_loss
        
        # Subject confusion loss (maximize entropy)
        if 'subject_logits' in outputs and 'subject' in targets:
            subject_loss = self.subject_criterion(
                outputs['subject_logits'],
                targets['subject']
            )
            losses['subject'] = subject_loss
        else:
            losses['subject'] = torch.tensor(0.0)
            
        # Consistency loss for mixup
        if mixed_outputs is not None and self.consistency_weight > 0:
            consistency_loss = F.mse_loss(
                outputs['features'],
                mixed_outputs['features'].detach()
            )
            losses['consistency'] = consistency_loss
        else:
            losses['consistency'] = torch.tensor(0.0)
            
        # Total loss
        losses['total'] = (
            self.task_weight * losses['task'] +
            self.subject_weight * losses['subject'] +
            self.consistency_weight * losses['consistency']
        )
        
        return losses


def create_subject_invariant_model(
    base_model: nn.Module,
    n_subjects: int,
    feature_dim: int = 64,
    lambda_reversal: float = 0.1,
    use_instance_norm: bool = True
) -> nn.Module:
    """
    Wrap base model with subject-invariant components
    """
    # Add instance normalization to feature extractor if requested
    if use_instance_norm and hasattr(base_model, 'feature_extractor'):
        # Insert instance norm as first layer
        original_temporal = base_model.feature_extractor.temporal_conv
        base_model.feature_extractor.temporal_conv = nn.Sequential(
            InstanceNormPerChannel(129),  # Assuming 129 channels
            original_temporal
        )
        
    # Create subject-invariant wrapper
    model = SubjectInvariantEEGENN(
        base_model=base_model,
        n_subjects=n_subjects,
        feature_dim=feature_dim,
        lambda_reversal=lambda_reversal
    )
    
    return model