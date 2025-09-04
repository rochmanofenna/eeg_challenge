"""
BICEP EEG Module - Stochastic Multi-Future Simulator for EEG
Implements Ornstein-Uhlenbeck SDE with event jumps for ERP modeling
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List


class OUParams:
    """Parameters for Ornstein-Uhlenbeck process"""
    def __init__(self, theta: float = 2.0, mu0: float = 0.0, sigma: float = 0.5):
        self.theta = theta  # Mean reversion rate
        self.mu0 = mu0      # Long-term mean
        self.sigma = sigma  # Volatility


class EEGSDE(nn.Module):
    """
    EEG-specific SDE simulator using Ornstein-Uhlenbeck process
    Models both oscillatory dynamics (SSVEP) and event-related jumps (ERP)
    """
    
    def __init__(
        self,
        dt: float = 0.01,
        ou_params: Optional[OUParams] = None,
        jump_rate: float = 0.1,
        jump_scale: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        self.dt = dt
        self.ou = ou_params if ou_params else OUParams()
        self.jump_rate = jump_rate
        self.jump_scale = jump_scale
        self.device = device
        
        # Learnable parameters for adaptive SDE
        self.theta_scale = nn.Parameter(torch.tensor(1.0))
        self.sigma_scale = nn.Parameter(torch.tensor(1.0))
        
    @torch.no_grad()
    def simulate_paths(
        self, 
        x: torch.Tensor, 
        N_paths: int = 64,
        use_antithetic: bool = True
    ) -> torch.Tensor:
        """
        Simulate N stochastic paths from base EEG trial using Euler-Maruyama
        
        Args:
            x: Base EEG trial [B, C, T]
            N_paths: Number of paths to simulate
            use_antithetic: Use antithetic sampling for variance reduction
            
        Returns:
            Simulated paths [N_paths, B, C, T]
        """
        B, C, T = x.shape
        dt = torch.tensor(self.dt, device=x.device)
        
        # Adaptive parameters
        theta = self.ou.theta * self.theta_scale.abs()
        sigma = self.ou.sigma * self.sigma_scale.abs()
        mu0 = self.ou.mu0
        
        # Initialize paths
        if use_antithetic and N_paths % 2 == 0:
            # Antithetic sampling for variance reduction
            N_half = N_paths // 2
            x0 = x.unsqueeze(0).repeat(N_half, 1, 1, 1)
            
            # Generate base noise
            eps = torch.randn_like(x0)
            
            # Create antithetic pairs
            eps_full = torch.cat([eps, -eps], dim=0)
            
            # Euler-Maruyama with OU dynamics
            dW = torch.sqrt(dt) * eps_full * sigma
            drift = theta * (mu0 - x0.repeat(2, 1, 1, 1)) * dt
            paths = x.unsqueeze(0).repeat(N_paths, 1, 1, 1) + drift + dW
            
        else:
            x0 = x.unsqueeze(0).repeat(N_paths, 1, 1, 1)
            
            # Standard Euler-Maruyama
            eps = torch.randn_like(x0)
            dW = torch.sqrt(dt) * eps * sigma
            drift = theta * (mu0 - x0) * dt
            paths = x0 + drift + dW
        
        # Add event-related jumps (ERP simulation)
        if self.jump_rate > 0 and self.jump_scale > 0:
            # Poisson-distributed jumps
            jump_mask = torch.rand(N_paths, B, C, 1, device=x.device) < self.jump_rate
            jump_magnitude = self.jump_scale * torch.randn(N_paths, B, C, 1, device=x.device)
            
            # Random jump times
            jump_times = torch.randint(T//4, 3*T//4, (N_paths, B, C, 1), device=x.device)
            
            # Create jump series
            jump_series = torch.zeros_like(paths)
            for n in range(N_paths):
                for b in range(B):
                    for c in range(C):
                        if jump_mask[n, b, c]:
                            t = jump_times[n, b, c, 0]
                            # Gaussian-shaped ERP
                            t_grid = torch.arange(T, device=x.device).float()
                            erp_shape = torch.exp(-0.5 * ((t_grid - t) / 10)**2)
                            jump_series[n, b, c] += jump_magnitude[n, b, c, 0] * erp_shape
            
            paths = paths + jump_series
        
        return paths
    
    def compute_path_statistics(
        self, 
        paths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute statistics across simulated paths
        
        Returns:
            mean, std, quantiles of paths
        """
        mean = paths.mean(dim=0)
        std = paths.std(dim=0)
        
        # Compute quantiles for uncertainty bands
        quantiles = torch.quantile(paths, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=paths.device), dim=0)
        
        return mean, std, quantiles


class OscillatorySDEVariant(EEGSDE):
    """
    Extended SDE for SSVEP modeling with explicit frequency components
    """
    
    def __init__(
        self,
        dt: float = 0.01,
        frequencies: List[float] = [10.0, 15.0, 20.0],  # Hz
        **kwargs
    ):
        super().__init__(dt=dt, **kwargs)
        self.frequencies = frequencies
        
        # Learnable frequency modulation
        self.freq_weights = nn.Parameter(torch.ones(len(frequencies)))
        
    @torch.no_grad()
    def simulate_paths(
        self, 
        x: torch.Tensor, 
        N_paths: int = 64,
        use_antithetic: bool = True
    ) -> torch.Tensor:
        """
        Simulate with added oscillatory components for SSVEP
        """
        # Base OU paths
        paths = super().simulate_paths(x, N_paths, use_antithetic)
        
        B, C, T = x.shape
        
        # Add oscillatory components
        for i, freq in enumerate(self.frequencies):
            t_grid = torch.arange(T, device=x.device).float() * self.dt
            
            # Random phase for each path/channel
            phases = torch.rand(N_paths, B, C, 1, device=x.device) * 2 * np.pi
            
            # Oscillation with frequency modulation
            oscillation = self.freq_weights[i].abs() * torch.sin(
                2 * np.pi * freq * t_grid.view(1, 1, 1, -1) + phases
            )
            
            # Add with random amplitude modulation
            amp_mod = 0.1 * torch.randn(N_paths, B, C, 1, device=x.device).abs()
            paths = paths + amp_mod * oscillation
        
        return paths


class AdaptiveBICEP(nn.Module):
    """
    Adaptive BICEP that learns SDE parameters from data
    """
    
    def __init__(
        self,
        base_sde: Optional[EEGSDE] = None,
        n_ensemble: int = 5
    ):
        super().__init__()
        
        if base_sde is None:
            base_sde = OscillatorySDEVariant()
        
        # Ensemble of SDEs with different parameters
        self.sde_ensemble = nn.ModuleList([
            OscillatorySDEVariant(
                theta=2.0 + 0.5 * i,
                sigma=0.5 + 0.1 * i
            ) for i in range(n_ensemble)
        ])
        
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(n_ensemble) / n_ensemble)
        
    def forward(
        self, 
        x: torch.Tensor,
        N_paths_per_sde: int = 20
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate ensemble of paths with learned weighting
        
        Returns:
            Weighted paths and uncertainty estimates
        """
        all_paths = []
        weights = torch.softmax(self.ensemble_weights, dim=0)
        
        for i, sde in enumerate(self.sde_ensemble):
            paths = sde.simulate_paths(x, N_paths=N_paths_per_sde)
            all_paths.append(paths * weights[i])
        
        # Combine weighted paths
        combined_paths = torch.cat(all_paths, dim=0)
        
        # Compute uncertainty as path variance
        path_mean = combined_paths.mean(dim=0)
        path_var = combined_paths.var(dim=0)
        
        return combined_paths, path_var
    
    def get_calibrated_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100
    ) -> torch.Tensor:
        """
        Get calibrated uncertainty estimates via Monte Carlo
        """
        uncertainties = []
        
        for _ in range(n_samples // 20):
            _, var = self.forward(x, N_paths_per_sde=20)
            uncertainties.append(var)
        
        # Average uncertainties
        return torch.stack(uncertainties).mean(dim=0)