"""
BEF Pipeline: Unified BICEP → ENN → Fusion Alpha for EEG
Complete implementation with uncertainty quantification and transfer learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np

from bicep_eeg import EEGSDE, OscillatorySDEVariant, AdaptiveBICEP
from enn import ENNEncoder, MultiScaleENN
from fusion_alpha import FusionAlphaGNN, HierarchicalFusionAlpha, build_sensor_graph


class BEF_EEG(nn.Module):
    """
    Complete BEF pipeline for EEG decoding
    BICEP: Multi-future stochastic simulation
    ENN: Entangled multi-state encoding
    Fusion Alpha: Graph-based contradiction resolution
    """
    
    def __init__(
        self,
        # Data parameters
        in_chans: int = 129,
        sfreq: int = 100,
        
        # BICEP parameters
        n_paths: int = 64,
        use_oscillatory_sde: bool = True,
        sde_frequencies: List[float] = [10.0, 15.0, 20.0],
        
        # ENN parameters  
        K: int = 16,
        embed_dim: int = 64,
        enn_layers: int = 2,
        use_multiscale: bool = False,
        
        # Fusion Alpha parameters
        gnn_hidden: int = 64,
        gnn_layers: int = 3,
        use_hierarchical: bool = False,
        
        # Output parameters
        output_dim: int = 1,
        
        # Training parameters
        dropout: float = 0.2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        self.n_paths = n_paths
        self.K = K
        self.device = device
        
        # BICEP: Stochastic simulator
        if use_oscillatory_sde:
            self.bicep = OscillatorySDEVariant(
                dt=1/sfreq,
                frequencies=sde_frequencies,
                device=device
            )
        else:
            self.bicep = EEGSDE(dt=1/sfreq, device=device)
        
        # ENN: Entangled encoder
        if use_multiscale:
            self.enn = MultiScaleENN(
                in_chans=in_chans,
                sfreq=sfreq,
                embed_dim=embed_dim,
                K=K,
                output_dim=output_dim,
                n_layers=enn_layers,
                scales=[1, 2, 4]
            )
        else:
            self.enn = ENNEncoder(
                in_chans=in_chans,
                sfreq=sfreq,
                embed_dim=embed_dim,
                K=K,
                output_dim=output_dim,
                n_layers=enn_layers
            )
        
        # Fusion Alpha: Graph fusion
        if use_hierarchical:
            self.fusion = HierarchicalFusionAlpha(
                node_feat_dim=K + 2,  # K states + mean/std from BICEP
                hidden_dim=gnn_hidden,
                output_dim=output_dim,
                n_layers=gnn_layers,
                dropout=dropout
            )
        else:
            self.fusion = FusionAlphaGNN(
                node_feat_dim=K + 2,
                hidden_dim=gnn_hidden,
                output_dim=output_dim,
                n_layers=gnn_layers,
                dropout=dropout
            )
        
        # Task-specific heads
        self.use_multi_task = False
        
    def extract_channel_features(
        self,
        enn_Z: torch.Tensor,
        enn_alpha: torch.Tensor,
        bicep_stats: Tuple[torch.Tensor, torch.Tensor],
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract per-channel features for Fusion Alpha nodes
        
        Args:
            enn_Z: ENN latent states [B, K]
            enn_alpha: ENN attention weights [B, K] 
            bicep_stats: (mean, std) from BICEP paths [B, C, T]
            x: Original EEG [B, C, T]
            
        Returns:
            Node features [B, C, K+2]
        """
        B, C, T = x.shape
        
        # Get BICEP statistics per channel
        bicep_mean, bicep_std = bicep_stats
        channel_mean = bicep_mean.mean(dim=-1)  # [B, C]
        channel_std = bicep_std.mean(dim=-1)    # [B, C]
        
        # Project ENN states to channels
        # Simple approach: broadcast ENN state to all channels
        # (Could be improved with learned projection)
        enn_broadcast = enn_Z.unsqueeze(1).expand(B, C, self.K)  # [B, C, K]
        
        # Weight by attention for different channels
        # (Heuristic: channels with higher variance get more uncertain states)
        channel_uncertainty = channel_std / (channel_std.mean(dim=1, keepdim=True) + 1e-6)
        weighted_enn = enn_broadcast * channel_uncertainty.unsqueeze(-1)
        
        # Concatenate features
        node_features = torch.cat([
            weighted_enn,                    # [B, C, K]
            channel_mean.unsqueeze(-1),      # [B, C, 1]
            channel_std.unsqueeze(-1)        # [B, C, 1]
        ], dim=-1)  # [B, C, K+2]
        
        return node_features
    
    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
        mc_samples: int = 4
    ) -> Dict[str, torch.Tensor]:
        """
        Full BEF forward pass
        
        Args:
            x: EEG data [B, C, T]
            return_intermediates: Return intermediate representations
            mc_samples: MC dropout samples for uncertainty
            
        Returns:
            Dictionary with predictions and uncertainties
        """
        B, C, T = x.shape
        
        # Stage 1: BICEP - Generate stochastic futures
        with torch.no_grad():
            paths = self.bicep.simulate_paths(x, N_paths=self.n_paths)  # [N, B, C, T]
        
        # Compute path statistics
        path_mean = paths.mean(dim=0)  # [B, C, T]
        path_std = paths.std(dim=0)    # [B, C, T]
        
        # Stage 2: ENN - Process through entangled network
        # We process the mean path (could also process multiple paths)
        enn_Z, enn_alpha, trajectory = self.enn(path_mean, return_trajectory=True)
        
        # Also get ENN output and uncertainty
        enn_output, enn_entropy = self.enn.get_output(path_mean)
        
        # Stage 3: Fusion Alpha - Graph-based fusion
        # Build sensor graph
        A = build_sensor_graph(x, k=8, use_correlation=True)
        
        # Extract node features
        node_features = self.extract_channel_features(
            enn_Z, enn_alpha, (path_mean, path_std), x
        )
        
        # Graph fusion with MC dropout
        fusion_out = self.fusion(
            node_features, A, 
            mc_samples=mc_samples
        )
        
        # Combine outputs
        results = {
            'prediction': fusion_out['logits'],
            'epistemic_uncertainty': fusion_out['uncertainty'],
            'aleatoric_uncertainty': enn_entropy,
            'total_uncertainty': fusion_out['uncertainty'] + enn_entropy,
            'attention_weights': fusion_out['attention']
        }
        
        if return_intermediates:
            results.update({
                'bicep_mean': path_mean,
                'bicep_std': path_std,
                'enn_Z': enn_Z,
                'enn_alpha': enn_alpha,
                'enn_trajectory': trajectory,
                'graph_adjacency': A
            })
        
        return results
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        task: str = "regression"
    ) -> Dict[str, torch.Tensor]:
        """
        Compute BEF loss with regularization
        """
        losses = {}
        
        if task == "regression":
            # Main regression loss
            pred = outputs['prediction'].squeeze(-1)
            losses['mse'] = F.mse_loss(pred, targets)
            
            # Uncertainty-weighted loss (optional)
            uncertainty = outputs['total_uncertainty'].squeeze(-1)
            weighted_mse = (pred - targets)**2 / (2 * uncertainty) + 0.5 * torch.log(uncertainty)
            losses['nll'] = weighted_mse.mean()
            
        elif task == "classification":
            # Binary classification
            logits = outputs['prediction'].squeeze(-1)
            losses['bce'] = F.binary_cross_entropy_with_logits(logits, targets.float())
        
        # Regularization
        losses['enn_reg'] = self.enn.regularization_loss(outputs.get('enn_alpha'))
        
        # Total loss
        losses['total'] = losses.get('mse', 0) + losses.get('bce', 0) + 0.01 * losses['enn_reg']
        
        return losses
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with calibrated uncertainty
        """
        predictions = []
        
        # Multiple forward passes
        for _ in range(n_samples // self.n_paths):
            out = self.forward(x, mc_samples=1)
            predictions.append(out['prediction'])
        
        predictions = torch.stack(predictions)
        
        # Compute mean and uncertainty
        mean_pred = predictions.mean(dim=0)
        total_uncertainty = predictions.var(dim=0)
        
        return mean_pred, total_uncertainty


class MultiTaskBEF(BEF_EEG):
    """
    Multi-task BEF for Challenge 1 & 2
    """
    
    def __init__(
        self,
        n_psycho_factors: int = 4,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.use_multi_task = True
        
        # Additional task heads
        self.rt_head = nn.Linear(kwargs.get('gnn_hidden', 64), 1)
        self.success_head = nn.Linear(kwargs.get('gnn_hidden', 64), 1)
        
        # Psychopathology heads (Challenge 2)
        self.psycho_heads = nn.ModuleList([
            nn.Linear(kwargs.get('gnn_hidden', 64), 1)
            for _ in range(n_psycho_factors)
        ])
        
    def forward_multitask(
        self,
        x: torch.Tensor,
        task: str = "all"
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-task forward pass
        """
        # Get base BEF outputs
        base_out = self.forward(x, return_intermediates=True)
        
        # Extract fusion embeddings
        fusion_embedding = base_out.get('fusion_embedding')
        
        outputs = {}
        
        if task in ["challenge1", "all"]:
            # Reaction time regression
            outputs['rt_prediction'] = self.rt_head(fusion_embedding)
            outputs['rt_uncertainty'] = base_out['total_uncertainty']
            
            # Success/failure classification
            outputs['success_logits'] = self.success_head(fusion_embedding)
            
        if task in ["challenge2", "all"]:
            # Psychopathology factors
            psycho_preds = []
            for head in self.psycho_heads:
                psycho_preds.append(head(fusion_embedding))
            
            outputs['psycho_predictions'] = torch.cat(psycho_preds, dim=-1)
            outputs['psycho_uncertainty'] = base_out['total_uncertainty'].expand(-1, len(self.psycho_heads))
        
        return outputs


class PretrainableBEF(BEF_EEG):
    """
    BEF with pretraining capabilities
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Contrastive learning head for pretraining
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.K, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
    def compute_contrastive_loss(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """
        SimCLR-style contrastive loss for pretraining
        """
        # Project embeddings
        h1 = self.contrastive_head(z1)
        h2 = self.contrastive_head(z2)
        
        # Normalize
        h1 = F.normalize(h1, dim=-1)
        h2 = F.normalize(h2, dim=-1)
        
        # Compute similarity
        sim = torch.matmul(h1, h2.T) / temperature
        
        # Contrastive loss
        labels = torch.arange(h1.shape[0], device=h1.device)
        loss = F.cross_entropy(sim, labels)
        
        return loss
    
    def pretrain_forward(
        self,
        x: torch.Tensor,
        augment: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for pretraining
        """
        if augment:
            # Create two augmented views
            x1 = x + 0.1 * torch.randn_like(x)  # Noise augmentation
            x2 = x * (0.9 + 0.2 * torch.rand_like(x))  # Amplitude augmentation
        else:
            x1 = x2 = x
        
        # Get ENN embeddings for both views
        z1, _, _ = self.enn(x1)
        z2, _, _ = self.enn(x2)
        
        # Compute contrastive loss
        loss = self.compute_contrastive_loss(z1, z2)
        
        return {'contrastive_loss': loss, 'z1': z1, 'z2': z2}