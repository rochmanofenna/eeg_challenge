"""
Codabench submission interface - Model class
This version uses the no-args constructor pattern
"""

import torch
import torch.nn as nn
from pathlib import Path
import random
import numpy as np
import os

# Import the BEF pipeline
try:
    from .pipeline import BEF_EEG, MultiTaskBEF
except ImportError:
    from pipeline import BEF_EEG, MultiTaskBEF



class Model(nn.Module):
    """
    Submission model for EEG Foundation Challenge
    No-args constructor as required by Codabench
    """
    
    def __init__(self):
        super().__init__()

        # Set deterministic mode for reproducibility
        self._seed_everything(42)

        # Set thread limits for consistent timing
        torch.set_num_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"

        # Default configuration for challenge with guardrails
        # 129 channels, 2s window @ 100Hz = 200 samples
        config = {
            'in_chans': 129,
            'sfreq': 100,
            'n_paths': min(16, 32),  # Clamped for safety
            'K': min(16, 32),  # Clamped graph neighbors
            'embed_dim': min(64, 128),
            'gnn_hidden': min(64, 128),
            'gnn_layers': min(3, 5),  # Cap layers
            'use_hierarchical': False,  # Faster inference
            'dropout': 0.0  # Disable for eval
        }
        self.config = config
        
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
        torch.set_grad_enabled(False)
    
    def _load_weights(self):
        """Load pretrained weights if available"""
        
        base_dir = Path(__file__).resolve().parent
        weight_paths = [
            base_dir / 'weights_challenge_1.pt',
            base_dir / 'weights_challenge_2.pt',
            Path('./') / 'weights_challenge_1.pt',
            Path('./bef_eeg') / 'weights_challenge_1.pt',
            Path('./') / 'weights_challenge_2.pt',
            Path('./bef_eeg') / 'weights_challenge_2.pt'
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
                    # Silent success (no prints in submission)
                    break

                except Exception as e:
                    # Silent failure
                    pass
    
    def _seed_everything(self, seed: int):
        """Set all seeds for deterministic behavior"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.use_deterministic_algorithms(True, warn_only=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for challenge evaluation

        Args:
            x: EEG data [B, 129, 200]

        Returns:
            predictions: [B, 1] for regression or [B, num_classes] for classification
        """
        # Input validation and shape normalization
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension if missing

        # Ensure contiguous memory layout
        x = x.contiguous()

        # Device-aware inference with optional AMP
        device = next(self.net.parameters()).device
        x = x.to(device)

        # Use AMP on CUDA only
        if device.type == 'cuda' and torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                with torch.inference_mode():
                    outputs = self.net(x, mc_samples=1)  # Single sample for speed
                    predictions = outputs['prediction']
        else:
            with torch.inference_mode():
                outputs = self.net(x, mc_samples=1)
                predictions = outputs['prediction']

        return predictions
    
    def predict(self, x: torch.Tensor) -> dict:
        """
        Main prediction interface with uncertainty quantification

        Returns:
            dict with 'logits', 'aleatoric', 'epistemic' uncertainty
        """
        # Input validation
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x = x.contiguous()
        device = next(self.net.parameters()).device
        x = x.to(device)

        # Adaptive compute: reduce paths if input is large
        B, C, T = x.shape
        budget = 2e7  # Element budget
        n_paths = self.config['n_paths']

        if B * C * T * n_paths > budget:
            n_paths = max(1, int(budget / (B * C * T)))

        with torch.inference_mode():
            # Get outputs with path-based uncertainty
            outputs = self.net(x, mc_samples=n_paths)

            predictions = outputs['prediction']

            # Split uncertainty (if available)
            if 'aleatoric_uncertainty' in outputs:
                aleatoric = outputs['aleatoric_uncertainty']
                epistemic = outputs['epistemic_uncertainty']
            else:
                # Estimate from total uncertainty
                total_unc = outputs.get('total_uncertainty', torch.zeros_like(predictions))
                # Heuristic split: 70% aleatoric, 30% epistemic
                aleatoric = 0.7 * total_unc
                epistemic = 0.3 * total_unc

        return {
            'logits': predictions,
            'aleatoric': aleatoric,
            'epistemic': epistemic
        }