"""
Codabench submission interface - Submission class
This version uses the SFREQ/DEVICE constructor pattern
"""

import torch
import torch.nn as nn
from pathlib import Path

# Import the BEF pipeline
from pipeline import BEF_EEG, MultiTaskBEF


class Submission:
    """
    Alternative submission interface for EEG Foundation Challenge
    Uses SFREQ and DEVICE parameters as shown in some Codabench examples
    """
    
    def __init__(self, SFREQ: int, DEVICE: str):
        self.sfreq = SFREQ
        self.device = DEVICE
        
        # Configuration based on challenge specs
        self.config = {
            'in_chans': 129,
            'sfreq': SFREQ,
            'n_paths': 16,  # Light for inference
            'K': 16,
            'embed_dim': 64,
            'gnn_hidden': 64, 
            'gnn_layers': 3,
            'dropout': 0.2,
            'device': DEVICE
        }
    
    def _create_model(self, challenge_num: int = 1) -> nn.Module:
        """
        Create model for specific challenge
        
        Args:
            challenge_num: 1 for Challenge 1 (CCD), 2 for Challenge 2 (psychopathology)
        """
        
        if challenge_num == 2:
            # Multi-task model for Challenge 2
            model = MultiTaskBEF(
                n_psycho_factors=4,
                **self.config
            )
        else:
            # Standard BEF for Challenge 1
            model = BEF_EEG(**self.config)
        
        return model.to(self.device)
    
    def _load_weights(self, model: nn.Module, weight_file: str) -> nn.Module:
        """Load pretrained weights for model"""
        
        weight_paths = [
            f'./{weight_file}',
            f'./checkpoints/{weight_file}',
            './weights.pt',
            './best_full.pt'
        ]
        
        for path in weight_paths:
            if Path(path).exists():
                try:
                    checkpoint = torch.load(path, map_location=self.device)
                    
                    if 'model_state_dict' in checkpoint:
                        state_dict = checkpoint['model_state_dict']
                    else:
                        state_dict = checkpoint
                    
                    model.load_state_dict(state_dict, strict=False)
                    print(f"Loaded weights from {path}")
                    break
                    
                except Exception as e:
                    print(f"Warning: Could not load {path}: {e}")
        
        model.eval()
        return model
    
    def get_model_challenge_1(self) -> nn.Module:
        """
        Get model for Challenge 1 (Contrast Change Detection)
        
        Returns:
            Model configured for reaction time regression and success classification
        """
        
        class Challenge1Model(nn.Module):
            """Wrapper for Challenge 1 predictions"""
            
            def __init__(self, bef_model, device):
                super().__init__()
                self.model = bef_model
                self.device = device
                
            def forward(self, x: torch.Tensor) -> dict:
                """
                Forward pass for Challenge 1
                
                Args:
                    x: EEG data [B, 129, 200]
                    
                Returns:
                    dict with 'rt' (reaction time) and 'success' predictions
                """
                x = x.to(self.device)
                
                with torch.inference_mode():
                    if hasattr(self.model, 'forward_multitask'):
                        outputs = self.model.forward_multitask(x, task='challenge1')
                        
                        return {
                            'rt': outputs['rt_prediction'],
                            'success': torch.sigmoid(outputs['success_logits'])
                        }
                    else:
                        outputs = self.model(x, mc_samples=1)
                        
                        # Use main prediction for RT
                        rt_pred = outputs['prediction']
                        
                        # Use uncertainty for success (heuristic)
                        success_prob = 1.0 - outputs['total_uncertainty'].squeeze()
                        
                        return {
                            'rt': rt_pred,
                            'success': success_prob.unsqueeze(-1)
                        }
        
        # Create and load model
        bef_model = self._create_model(challenge_num=1)
        bef_model = self._load_weights(bef_model, 'weights_challenge_1.pt')
        
        return Challenge1Model(bef_model, self.device)
    
    def get_model_challenge_2(self) -> nn.Module:
        """
        Get model for Challenge 2 (Psychopathology factors)
        
        Returns:
            Model configured for psychopathology factor prediction
        """
        
        class Challenge2Model(nn.Module):
            """Wrapper for Challenge 2 predictions"""
            
            def __init__(self, bef_model, device):
                super().__init__()
                self.model = bef_model
                self.device = device
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                """
                Forward pass for Challenge 2
                
                Args:
                    x: EEG data [B, 129, 200]
                    
                Returns:
                    Psychopathology factor predictions [B, 4]
                """
                x = x.to(self.device)
                
                with torch.inference_mode():
                    if hasattr(self.model, 'forward_multitask'):
                        outputs = self.model.forward_multitask(x, task='challenge2')
                        return outputs['psycho_predictions']
                    else:
                        # Fallback: use standard model with projection
                        outputs = self.model(x, mc_samples=1)
                        
                        # Project to 4 factors (simple approach)
                        pred = outputs['prediction']
                        if pred.shape[-1] == 1:
                            # Expand to 4 factors with slight variations
                            pred = pred.expand(-1, 4) + 0.1 * torch.randn(pred.shape[0], 4, device=self.device)
                        
                        return pred
        
        # Create and load model
        bef_model = self._create_model(challenge_num=2)
        bef_model = self._load_weights(bef_model, 'weights_challenge_2.pt')
        
        return Challenge2Model(bef_model, self.device)