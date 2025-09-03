"""
Submission file for NeurIPS 2025 EEG Challenge
This file will be evaluated on Codabench
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CompactEEGENN(nn.Module):
    """Compact EEG model with ENN-inspired design for submission"""
    
    def __init__(self, n_chans: int = 129, n_times: int = 200, sfreq: int = 100):
        super().__init__()
        
        # Temporal convolution (proven for EEG)
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), padding=(0, 12)),
            nn.BatchNorm2d(40),
            nn.ELU()
        )
        
        # Spatial convolution
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(40, 40, (n_chans, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.25)
        )
        
        # Separable convolutions
        self.separable = nn.Sequential(
            nn.Conv2d(40, 40, (1, 15), padding=(0, 7), groups=40),
            nn.Conv2d(40, 80, (1, 1)),
            nn.BatchNorm2d(80),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.25)
        )
        
        # Calculate flattened size
        # After convs: time = 200 -> 200/4 -> 50/8 = 6
        self.flatten_size = 80 * 6
        
        # ENN-inspired head with smooth activation
        self.head = nn.Sequential(
            nn.Linear(self.flatten_size, 64),
            nn.Tanh(),  # Smooth activation like ENN
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            x: EEG data [batch, channels, time]
        Returns:
            predictions [batch, 1]
        """
        # Ensure correct input shape
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [batch, 1, channels, time]
            
        # Feature extraction
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.separable(x)
        
        # Flatten and predict
        x = x.flatten(1)
        x = self.head(x)
        
        return x


class Submission:
    """
    Main submission class for EEG Challenge
    Must implement get_model_challenge_1 and get_model_challenge_2
    """
    
    def __init__(self, SFREQ: int, DEVICE: torch.device):
        self.sfreq = SFREQ
        self.device = DEVICE
        
        # Initialize models
        self.model1 = CompactEEGENN(n_chans=129, n_times=200, sfreq=self.sfreq)
        self.model2 = CompactEEGENN(n_chans=129, n_times=200, sfreq=self.sfreq)
        
        # Move to device
        self.model1 = self.model1.to(self.device)
        self.model2 = self.model2.to(self.device)
        
        # Load pretrained weights if available
        try:
            checkpoint1 = torch.load("weights_challenge_1.pt", map_location=self.device)
            self.model1.load_state_dict(checkpoint1, strict=False)
            print("Loaded weights for Challenge 1")
        except Exception as e:
            print(f"Could not load Challenge 1 weights: {e}")
            print("Using random initialization for Challenge 1")
            
        try:
            checkpoint2 = torch.load("weights_challenge_2.pt", map_location=self.device)
            self.model2.load_state_dict(checkpoint2, strict=False)
            print("Loaded weights for Challenge 2")
        except Exception as e:
            print(f"Could not load Challenge 2 weights: {e}")
            print("Using random initialization for Challenge 2")
            
        # Set models to eval mode
        self.model1.eval()
        self.model2.eval()
        
        # Ensure no gradients
        for param in self.model1.parameters():
            param.requires_grad = False
        for param in self.model2.parameters():
            param.requires_grad = False
            
    def get_model_challenge_1(self) -> nn.Module:
        """
        Get model for Challenge 1: Cross-Task Transfer Learning
        Predicts RT and success rate from CCD task
        """
        return self.model1
        
    def get_model_challenge_2(self) -> nn.Module:
        """
        Get model for Challenge 2: Psychopathology Factor Prediction
        Predicts 4 psychopathology factors
        """
        return self.model2


# Alternative implementation using braindecode models (if available in their docker)
class SubmissionWithBraindecode:
    """
    Alternative submission using braindecode models
    Uncomment if you want to use braindecode instead
    """
    
    def __init__(self, SFREQ: int, DEVICE: torch.device):
        self.sfreq = SFREQ
        self.device = DEVICE
        
        # Try to import braindecode
        try:
            from braindecode.models import EEGNetv4
            self.use_braindecode = True
        except ImportError:
            print("Braindecode not available, using custom model")
            self.use_braindecode = False
            
        if self.use_braindecode:
            # Use EEGNetv4 from braindecode
            self.model1 = EEGNetv4(
                in_chans=129,
                n_classes=1,
                input_window_samples=int(2 * self.sfreq)
            ).to(self.device)
            
            self.model2 = EEGNetv4(
                in_chans=129,
                n_classes=1,  # Will output 4 values through reshaping if needed
                input_window_samples=int(2 * self.sfreq)
            ).to(self.device)
        else:
            # Fallback to custom model
            self.model1 = CompactEEGENN().to(self.device)
            self.model2 = CompactEEGENN().to(self.device)
            
        # Load weights
        self._load_weights()
        
        # Set eval mode
        self.model1.eval()
        self.model2.eval()
        
    def _load_weights(self):
        """Load pretrained weights"""
        try:
            self.model1.load_state_dict(
                torch.load("weights_challenge_1.pt", map_location=self.device),
                strict=False
            )
        except:
            pass
            
        try:
            self.model2.load_state_dict(
                torch.load("weights_challenge_2.pt", map_location=self.device),
                strict=False
            )
        except:
            pass
            
    def get_model_challenge_1(self) -> nn.Module:
        return self.model1
        
    def get_model_challenge_2(self) -> nn.Module:
        return self.model2