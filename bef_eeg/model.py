"""
Codabench submission interface - Model class
This version uses the no-args constructor pattern
"""

import torch
import torch.nn as nn
from pathlib import Path

# Import the BEF pipeline
from pipeline import BEF_EEG, MultiTaskBEF


class Model(nn.Module):
    """
    Submission model for EEG Foundation Challenge
    No-args constructor as required by Codabench
    """
    
    def __init__(self):
        super().__init__()
        
        # Default configuration for challenge
        # 129 channels, 2s window @ 100Hz = 200 samples
        config = {
            'in_chans': 129,
            'sfreq': 100,
            'n_paths': 32,  # Reduced for inference speed
            'K': 16,
            'embed_dim': 64,
            'gnn_hidden': 64,
            'gnn_layers': 3,
            'use_hierarchical': False,  # Faster inference
            'dropout': 0.2
        }
        
        # Initialize BEF model
        self.net = BEF_EEG(
            in_chans=config['in_chans'],
            sfreq=config['sfreq'],
            n_paths=config['n_paths'],
            K=config['K'],
            embed_dim=config['embed_dim'],
            gnn_hidden=config['gnn_hidden'],
            gnn_layers=config['gnn_layers'],
            use_hierarchical=config['use_hierarchical'],
            dropout=config['dropout']
        )
        
        # Try to load pretrained weights
        self._load_weights()
        
        # Set to eval mode
        self.eval()
    
    def _load_weights(self):
        """Load pretrained weights if available"""
        
        weight_paths = [
            './weights.pt',
            './best_full.pt',
            './checkpoints/best_full.pt'
        ]
        
        for path in weight_paths:
            if Path(path).exists():
                try:
                    checkpoint = torch.load(path, map_location='cpu')
                    
                    # Handle different checkpoint formats
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                    
                    # Load weights
                    self.net.load_state_dict(state_dict, strict=False)
                    print(f"Loaded weights from {path}")
                    break
                    
                except Exception as e:
                    print(f"Warning: Could not load {path}: {e}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for challenge evaluation
        
        Args:
            x: EEG data [B, 129, 200] 
            
        Returns:
            predictions: [B, 1] for regression or [B, num_classes] for classification
        """
        # Ensure correct dimensions
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing
        
        # Get BEF predictions with reduced MC samples for speed
        with torch.no_grad():
            outputs = self.net(x, mc_samples=4)
            predictions = outputs['prediction']
        
        return predictions
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> tuple:
        """
        Optional: Predictions with uncertainty for better calibration
        """
        with torch.no_grad():
            outputs = self.net(x, mc_samples=10)
            
            predictions = outputs['prediction']
            uncertainty = outputs['total_uncertainty']
            
        return predictions, uncertainty