"""
ENN-based EEG model for NeurIPS 2025 EEG Challenge
Combines EEG-specific backbone with ENN uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class ENNCell(nn.Module):
    """ENN-style cell with PSD constraint and uncertainty estimation"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Initialize L for E = L @ L.T (ensuring PSD)
        self.L = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(hidden_dim))
        
        # Output projections
        self.W_mean = nn.Linear(hidden_dim, output_dim)
        self.W_log_var = nn.Linear(hidden_dim, output_dim)
        
        # Regularization strength
        self.register_buffer('reg_strength', torch.tensor(1e-4))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation
        
        Args:
            x: Input tensor [batch, input_dim]
            
        Returns:
            mean: Predicted mean [batch, output_dim]
            log_var: Log variance for uncertainty [batch, output_dim]
        """
        # Compute hidden state with PSD constraint
        h = x @ self.L + self.b
        h = torch.tanh(h)  # Smooth activation
        
        # Compute outputs
        mean = self.W_mean(h)
        log_var = self.W_log_var(h)
        
        # Ensure numerical stability
        log_var = torch.clamp(log_var, min=-10, max=2)
        
        return mean, log_var
    
    def get_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """Get calibrated uncertainty estimates"""
        _, log_var = self.forward(x)
        return torch.exp(0.5 * log_var)
    
    def regularization_loss(self) -> torch.Tensor:
        """Compute regularization to maintain well-conditioned E matrix"""
        E = self.L @ self.L.T
        # Encourage diagonal dominance
        diag_sum = torch.diagonal(E).sum()
        total_sum = E.sum()
        reg = self.reg_strength * (total_sum - diag_sum).abs()
        return reg


class TemporalBlock(nn.Module):
    """Temporal convolution block for EEG sequence modeling"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2
        )
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ELU()
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, channels, time]"""
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        return out


class EEGFeatureExtractor(nn.Module):
    """
    EEG-specific feature extractor
    Handles 129 channels x 200 timepoints (2s @ 100Hz)
    """
    
    def __init__(self, n_chans: int = 129, n_times: int = 200, n_filters: int = 40):
        super().__init__()
        
        # Temporal convolution (extract temporal patterns)
        self.temporal_conv = nn.Conv2d(1, n_filters, (1, 25), padding=(0, 12))
        
        # Spatial convolution (combine channels)
        self.spatial_conv = nn.Conv2d(n_filters, n_filters, (n_chans, 1))
        self.norm1 = nn.BatchNorm2d(n_filters)
        
        # Separable convolutions for efficiency
        self.separable = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, (1, 15), padding=(0, 7), groups=n_filters),
            nn.Conv2d(n_filters, n_filters * 2, (1, 1)),
            nn.BatchNorm2d(n_filters * 2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25)
        )
        
        # Calculate output size (more precise calculation)
        # After temporal conv (padding=12): n_times stays 200
        # After first pool (8): 200 / 8 = 25
        # After second pool (8): 25 / 8 = 3 (but AvgPool2d rounds down)
        # Actually let's measure it dynamically
        with torch.no_grad():
            test_input = torch.randn(1, 1, n_chans, n_times)
            temp_out = self.temporal_conv(test_input)
            spatial_out = self.spatial_conv(temp_out)
            sep_out = self.separable(spatial_out)
            self.feature_dim = sep_out.flatten(1).shape[1]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: EEG data [batch, channels, time]
            
        Returns:
            features: Extracted features [batch, feature_dim]
        """
        # Add channel dimension for 2D convs
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [batch, 1, channels, time]
            
        # Temporal filtering
        out = self.temporal_conv(x)
        
        # Spatial filtering
        out = self.spatial_conv(out)
        out = self.norm1(out)
        out = F.elu(out)
        
        # Separable convolutions
        out = self.separable(out)
        
        # Flatten
        out = out.flatten(1)
        
        return out


class EEGEANN(nn.Module):
    """
    Complete EEG + ENN model for the challenge
    """
    
    def __init__(
        self,
        n_chans: int = 129,
        n_times: int = 200,
        hidden_dim: int = 64,
        output_dim: int = 1,
        n_filters: int = 40
    ):
        super().__init__()
        
        # EEG feature extraction
        self.feature_extractor = EEGFeatureExtractor(n_chans, n_times, n_filters)
        
        # Dimension reduction
        self.projection = nn.Sequential(
            nn.Linear(self.feature_extractor.feature_dim, 128),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64)
        )
        
        # ENN cell for final prediction with uncertainty
        self.enn_cell = ENNCell(64, hidden_dim, output_dim)
        
        # Task-specific heads (for multi-task learning)
        self.use_multi_task = False
        
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        return_uncertainty: bool = True
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        """
        Forward pass
        
        Args:
            x: EEG data [batch, channels, time]
            return_features: Whether to return intermediate features
            return_uncertainty: Whether to return uncertainty estimates
            
        Returns:
            If return_uncertainty is False: predictions [batch, output_dim]
            If return_uncertainty is True: (predictions, uncertainty) tuple
            If return_features is True: Also includes features
        """
        # Extract EEG features
        features = self.feature_extractor(x)
        
        # Project to ENN input dimension
        projected = self.projection(features)
        
        # ENN forward pass
        mean, log_var = self.enn_cell(projected)
        
        outputs = [mean]
        
        if return_uncertainty:
            uncertainty = torch.exp(0.5 * log_var)
            outputs.append(uncertainty)
            
        if return_features:
            outputs.append(projected)
            
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
    
    def get_regularization_loss(self) -> torch.Tensor:
        """Get total regularization loss"""
        return self.enn_cell.regularization_loss()


class MultiTaskEEGENN(EEGEANN):
    """
    Extended model for multi-task learning (Challenge 1 & 2)
    """
    
    def __init__(self, *args, n_psychopathology_factors: int = 4, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.use_multi_task = True
        
        # Additional heads for psychopathology factors (Challenge 2)
        self.psychopathology_heads = nn.ModuleList([
            ENNCell(64, 32, 1) for _ in range(n_psychopathology_factors)
        ])
        
        # Task routing
        self.task_router = nn.Linear(64, 2 + n_psychopathology_factors)
        
    def forward_multitask(
        self,
        x: torch.Tensor,
        task: str = "all"
    ) -> dict:
        """
        Multi-task forward pass
        
        Args:
            x: EEG data [batch, channels, time]
            task: Which task(s) to compute ("challenge1", "challenge2", "all")
            
        Returns:
            Dictionary with task outputs
        """
        # Extract shared features
        features = self.feature_extractor(x)
        projected = self.projection(features)
        
        outputs = {}
        
        if task in ["challenge1", "all"]:
            # RT and success prediction
            mean, log_var = self.enn_cell(projected)
            outputs["rt_mean"] = mean
            outputs["rt_uncertainty"] = torch.exp(0.5 * log_var)
            
        if task in ["challenge2", "all"]:
            # Psychopathology factors
            psycho_outputs = []
            psycho_uncertainties = []
            
            for head in self.psychopathology_heads:
                mean, log_var = head(projected)
                psycho_outputs.append(mean)
                psycho_uncertainties.append(torch.exp(0.5 * log_var))
                
            outputs["psychopathology"] = torch.cat(psycho_outputs, dim=1)
            outputs["psycho_uncertainty"] = torch.cat(psycho_uncertainties, dim=1)
            
        # Task attention weights (which task is this sample good for?)
        outputs["task_weights"] = F.softmax(self.task_router(projected), dim=1)
        
        return outputs