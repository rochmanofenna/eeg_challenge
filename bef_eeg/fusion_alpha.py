"""
Fusion Alpha: Graph-based decision fusion with contradiction resolution
Implements GNN for EEG channel/feature fusion with uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict
import numpy as np
import hashlib
import time


class GraphCache:
    """Cache for adjacency matrices to avoid recomputation"""

    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def _make_key(self, C: int, k: int, use_corr: bool, use_spatial: bool, positions_hash: Optional[str] = None):
        """Create cache key from parameters"""
        return (C, k, use_corr, use_spatial, positions_hash)

    def get(self, key):
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            # LRU eviction: remove oldest entry
            oldest = next(iter(self.cache))
            del self.cache[oldest]
        self.cache[key] = value

    def clear(self):
        self.cache.clear()
        self.hits = 0
        self.misses = 0


# Global cache instance
_graph_cache = GraphCache()


def build_sensor_graph(
    x: torch.Tensor,
    k: int = 8,
    use_correlation: bool = True,
    use_spatial: bool = True,
    channel_positions: Optional[torch.Tensor] = None,
    cache: Optional[GraphCache] = None
) -> torch.Tensor:
    """
    Build adjacency matrix for EEG sensors with optional caching

    Args:
        x: EEG data [B, C, T]
        k: Number of neighbors for k-NN
        use_correlation: Use temporal correlation for edges
        use_spatial: Use spatial proximity for edges
        channel_positions: Optional 3D positions of channels [C, 3]
        cache: Optional GraphCache for reusing adjacency matrices

    Returns:
        A: Adjacency matrix [B, C, C]
    """
    B, C, T = x.shape
    device = x.device

    # Use global cache if none provided
    if cache is None:
        cache = _graph_cache
    
    # Try to retrieve from cache if spatial-only (deterministic)
    if not use_correlation and use_spatial and channel_positions is not None:
        # Hash positions for cache key
        pos_bytes = channel_positions.cpu().numpy().tobytes()
        pos_hash = hashlib.md5(pos_bytes).hexdigest()
        cache_key = cache._make_key(C, k, False, True, pos_hash)

        cached = cache.get(cache_key)
        if cached is not None:
            # Broadcast cached adjacency to batch
            return cached.unsqueeze(0).expand(B, -1, -1).to(device)

    # Initialize adjacency
    A = torch.zeros(B, C, C, device=device)
    
    if use_correlation:
        # Compute correlation-based adjacency
        # Normalize signals
        x_norm = F.normalize(x, dim=2)  # [B, C, T]
        
        # Compute pairwise correlations
        corr = torch.einsum('bct,bdt->bcd', x_norm, x_norm) / T  # [B, C, C]
        
        # Keep top-k correlations per node (vectorized)
        topk_vals, topk_idx = torch.topk(corr.abs(), k=min(k, C), dim=-1)

        # Vectorized adjacency creation
        b_idx = torch.arange(B, device=device).view(-1, 1, 1)
        c_idx = torch.arange(C, device=device).view(1, -1, 1)
        A[b_idx, c_idx, topk_idx] = topk_vals.to(A.dtype)
    
    if use_spatial and channel_positions is not None:
        # Add spatial proximity edges
        # Compute pairwise distances
        dist = torch.cdist(channel_positions, channel_positions)  # [C, C]
        
        # Convert distances to similarities
        sigma = dist.mean()
        spatial_sim = torch.exp(-dist**2 / (2 * sigma**2))
        
        # Add to adjacency (broadcast to batch)
        A = A + 0.5 * spatial_sim.unsqueeze(0)
    
    # Prune tiny edges for sparsity
    threshold = 0.01
    A = A * (A > threshold).float()

    # Add self-loops
    I = torch.eye(C, device=device).unsqueeze(0)
    A = A + I

    # Normalize adjacency (symmetric normalization)
    D = A.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    D_inv_sqrt = torch.pow(D, -0.5)
    A_norm = D_inv_sqrt * A * D_inv_sqrt.transpose(-1, -2)

    # Cache if spatial-only (deterministic)
    if not use_correlation and use_spatial and channel_positions is not None:
        cache.set(cache_key, A_norm[0].cpu())  # Cache first in batch

    return A_norm


class GCNLayer(nn.Module):
    """Graph Convolutional Network layer"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bias: bool = True,
        activation: str = "elu",
        dropout: float = 0.0  # Default to 0 for inference
    ):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=use_bias)

        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        
    def forward(self, H: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: Node features [B, N, F_in]
            A: Adjacency matrix [B, N, N]
        Returns:
            Updated features [B, N, F_out]
        """
        # Graph convolution: H' = A H W
        H_agg = torch.einsum('bnm,bmf->bnf', A, H)  # [B, N, F_in]
        H_out = self.lin(H_agg)  # [B, N, F_out]
        H_out = self.activation(H_out)
        H_out = self.dropout(H_out)
        
        return H_out


class GraphAttentionLayer(nn.Module):
    """Graph Attention Network layer"""
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_heads: int = 4,
        concat: bool = True,
        dropout: float = 0.0  # Default to 0 for inference
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.concat = concat
        
        # Multi-head projections
        self.W = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        
        # Attention parameters
        self.a = nn.Parameter(torch.randn(n_heads, 2 * out_dim))
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(
        self,
        H: torch.Tensor,
        A: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            H: Node features [B, N, F_in]
            A: Optional adjacency for masking [B, N, N]
        Returns:
            Updated features [B, N, F_out * n_heads] or [B, N, F_out]
        """
        B, N, _ = H.shape
        
        # Transform features
        H_transformed = self.W(H).view(B, N, self.n_heads, self.out_dim)
        H_transformed = H_transformed.permute(0, 2, 1, 3)  # [B, n_heads, N, out_dim]
        
        # Compute attention scores
        H_i = H_transformed.unsqueeze(3)  # [B, n_heads, N, 1, out_dim]
        H_j = H_transformed.unsqueeze(2)  # [B, n_heads, 1, N, out_dim]
        
        # Concatenate for attention
        H_cat = torch.cat([
            H_i.expand(-1, -1, -1, N, -1),
            H_j.expand(-1, -1, N, -1, -1)
        ], dim=-1)  # [B, n_heads, N, N, 2*out_dim]
        
        # Compute attention coefficients
        e = torch.einsum('bhnmd,hd->bhnm', H_cat, self.a)
        e = self.leaky_relu(e)
        
        # Mask if adjacency provided
        if A is not None:
            # Expand A for all heads
            A_expanded = A.unsqueeze(1)  # [B, 1, N, N]
            e = e.masked_fill(A_expanded == 0, -1e9)
        
        # Softmax
        alpha = F.softmax(e, dim=-1)
        alpha = self.dropout(alpha)
        
        # Apply attention
        H_attended = torch.einsum('bhnm,bhmd->bhnd', alpha, H_transformed)
        
        if self.concat:
            # Concatenate heads
            H_out = H_attended.permute(0, 2, 1, 3).contiguous()
            H_out = H_out.view(B, N, -1)  # [B, N, out_dim * n_heads]
        else:
            # Average heads
            H_out = H_attended.mean(dim=1)  # [B, N, out_dim]
            
        return F.elu(H_out)


class FusionAlphaGNN(nn.Module):
    """
    Graph Neural Network for Fusion Alpha
    Resolves contradictions and fuses multi-modal evidence
    """
    
    def __init__(
        self,
        node_feat_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        n_layers: int = 3,
        use_attention: bool = True,
        dropout: float = 0.0,  # Default to 0 for inference
        max_time: float = 2.0,  # Maximum time in seconds
        adaptive_layers: bool = True  # Enable early exit
    ):
        super().__init__()

        self.n_layers = min(n_layers, 5)  # Cap layers
        self.use_attention = use_attention
        self.dropout_rate = dropout
        self.max_time = max_time
        self.adaptive_layers = adaptive_layers
        
        # Build GNN layers with mixed architecture for efficiency
        self.gnn_layers = nn.ModuleList()

        for i in range(self.n_layers):
            in_dim = node_feat_dim if i == 0 else hidden_dim

            # Use GCN for early layers, GAT only for last layer if attention enabled
            if use_attention and i == self.n_layers - 1:
                # Only use attention on the last layer to save compute
                layer = GraphAttentionLayer(
                    in_dim=in_dim,
                    out_dim=hidden_dim,
                    n_heads=2,  # Reduced heads for speed
                    concat=False,
                    dropout=dropout
                )
            else:
                # Use efficient GCN for other layers
                layer = GCNLayer(in_dim=in_dim, out_dim=hidden_dim, dropout=dropout)

            self.gnn_layers.append(layer)

        # Early exit heads for adaptive computation
        if self.adaptive_layers:
            self.early_exit_heads = nn.ModuleList([
                nn.Linear(hidden_dim, output_dim)
                for _ in range(self.n_layers - 1)
            ])
        
        # Readout layers
        self.readout_attention = nn.Linear(hidden_dim, 1)
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # MC Dropout for uncertainty
        self.mc_dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        node_feats: torch.Tensor,
        A: torch.Tensor,
        mc_samples: int = 1,
        return_node_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through GNN with early exit capability

        Args:
            node_feats: Node features [B, N, F]
            A: Adjacency matrix [B, N, N]
            mc_samples: Number of MC dropout samples
            return_node_embeddings: Return intermediate embeddings

        Returns:
            Dictionary with 'logits', 'uncertainty', optional 'embeddings'
        """
        B, N, _ = node_feats.shape

        # Input validation and safety checks
        if B * N > 10000:  # Large batch/node check
            mc_samples = min(mc_samples, 1)  # Reduce MC samples for large inputs

        # Time tracking for early exit
        start_time = time.time()

        # Forward through GNN layers with early exit
        H = node_feats
        embeddings = []
        early_exit_triggered = False
        exit_layer = self.n_layers

        for i, layer in enumerate(self.gnn_layers):
            # Check time budget
            if self.adaptive_layers and i > 0:
                elapsed = time.time() - start_time
                if elapsed > self.max_time * 0.7:  # Use 70% of budget
                    early_exit_triggered = True
                    exit_layer = i
                    break

            # Forward through layer
            H = layer(H, A)
            embeddings.append(H)

            # Early exit prediction if available
            if self.adaptive_layers and i < self.n_layers - 1:
                if B * N * (self.n_layers - i) > 50000:  # Complexity check
                    # Use early exit head
                    early_exit_triggered = True
                    exit_layer = i + 1
                    break

        # Handle early exit if triggered
        if early_exit_triggered and self.adaptive_layers and exit_layer < self.n_layers:
            # Use simplified readout for early exit
            H_pooled = H.mean(dim=1)  # Simple mean pooling
            logits = self.early_exit_heads[exit_layer - 1](H_pooled)
            return {
                'logits': logits,
                'uncertainty': torch.zeros_like(logits),
                'attention': torch.ones(B, N, device=H.device) / N,  # Uniform attention
                'global_embedding': H_pooled,
                'early_exit': True,
                'exit_layer': exit_layer
            }

        # Normal path: Attention-based readout
        attention_scores = self.readout_attention(H)  # [B, N, 1]
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted aggregation
        H_global = torch.sum(attention_weights * H, dim=1)  # [B, hidden_dim]
        
        # MC Dropout for uncertainty (skip if inference mode)
        if self.training or mc_samples > 1:
            outputs = []
            for _ in range(mc_samples):
                H_drop = self.mc_dropout(H_global)
                out = self.readout_mlp(H_drop)
                outputs.append(out)

            outputs = torch.stack(outputs, dim=0)  # [mc_samples, B, output_dim]

            # Compute mean and uncertainty
            logits = outputs.mean(dim=0)  # [B, output_dim]
            uncertainty = outputs.var(dim=0) if mc_samples > 1 else torch.zeros_like(logits)
        else:
            # Fast inference path without MC dropout
            logits = self.readout_mlp(H_global)
            uncertainty = torch.zeros_like(logits)
        
        result = {
            'logits': logits,
            'uncertainty': uncertainty,
            'attention': attention_weights.squeeze(-1),
            'global_embedding': H_global,
            'early_exit': False,
            'exit_layer': self.n_layers
        }

        if return_node_embeddings:
            result['embeddings'] = embeddings

        return result


class ContradictionResolver(nn.Module):
    """
    Explicit contradiction detection and resolution module
    """
    
    def __init__(self, feat_dim: int, threshold: float = 0.5):
        super().__init__()
        self.threshold = threshold
        
        # Contradiction detector
        self.detector = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ELU(),
            nn.Linear(feat_dim, 1),
            nn.Sigmoid()
        )
        
        # Resolution network
        self.resolver = nn.Sequential(
            nn.Linear(feat_dim * 2 + 1, feat_dim),
            nn.ELU(),
            nn.Linear(feat_dim, feat_dim)
        )
        
    def forward(
        self,
        H: torch.Tensor,
        A: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect and resolve contradictions
        
        Returns:
            Resolved features and contradiction mask
        """
        B, N, F = H.shape
        
        # Vectorized contradiction detection for efficiency
        # Get indices of connected nodes
        connected_mask = A > 0.1  # [B, N, N]

        # For simplicity, compute pairwise contradictions only for strongly connected nodes
        contradictions = torch.zeros(B, N, N, device=H.device)

        # Efficient batch computation using broadcasting
        H_i = H.unsqueeze(2)  # [B, N, 1, F]
        H_j = H.unsqueeze(1)  # [B, 1, N, F]
        pair_features = torch.cat([H_i.expand(B, N, N, F), H_j.expand(B, N, N, F)], dim=-1)  # [B, N, N, 2F]

        # Only compute where connected
        pair_features = pair_features[connected_mask]
        if pair_features.numel() > 0:
            contra_scores = self.detector(pair_features).squeeze(-1)
            contradictions[connected_mask] = contra_scores
        
        # Identify contradictory nodes
        contra_mask = (contradictions.max(dim=-1)[0] > self.threshold)
        
        # Resolve contradictions
        H_resolved = H.clone()
        
        for b in range(B):
            for n in range(N):
                if contra_mask[b, n]:
                    # Get neighbors
                    neighbors = torch.where(A[b, n] > 0)[0]
                    if len(neighbors) > 0:
                        # Aggregate neighbor features
                        neighbor_feat = H[b, neighbors].mean(dim=0)
                        
                        # Resolve
                        concat_feat = torch.cat([
                            H[b, n],
                            neighbor_feat,
                            contradictions[b, n].max().unsqueeze(0)
                        ])
                        H_resolved[b, n] = self.resolver(concat_feat)
        
        return H_resolved, contra_mask


class HierarchicalFusionAlpha(nn.Module):
    """
    Hierarchical graph fusion: channel-level → region-level → global
    """
    
    def __init__(
        self,
        node_feat_dim: int,
        region_map: Optional[Dict[str, List[int]]] = None,
        **kwargs
    ):
        super().__init__()
        
        # Default brain regions for EEG
        if region_map is None:
            self.region_map = {
                'frontal': list(range(0, 30)),
                'central': list(range(30, 60)),
                'parietal': list(range(60, 90)),
                'occipital': list(range(90, 120)),
                'temporal': list(range(120, 129))
            }
        else:
            self.region_map = region_map
            
        self.n_regions = len(self.region_map)
        
        # Channel-level GNN
        self.channel_gnn = FusionAlphaGNN(node_feat_dim, **kwargs)
        
        # Region aggregator
        self.region_agg = nn.Linear(kwargs.get('hidden_dim', 64), kwargs.get('hidden_dim', 64))
        
        # Region-level GNN
        self.region_gnn = FusionAlphaGNN(
            node_feat_dim=kwargs.get('hidden_dim', 64),
            **kwargs
        )
        
    def forward(
        self,
        node_feats: torch.Tensor,
        A: torch.Tensor,
        mc_samples: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Hierarchical forward pass
        """
        B = node_feats.shape[0]
        device = node_feats.device
        
        # Channel-level processing
        channel_out = self.channel_gnn(node_feats, A, mc_samples=1, return_node_embeddings=True)
        channel_embeddings = channel_out['embeddings'][-1]  # [B, N_channels, hidden]
        
        # Aggregate to regions
        region_feats = []
        for region_name, channel_ids in self.region_map.items():
            if channel_ids:
                region_feat = channel_embeddings[:, channel_ids].mean(dim=1)
                region_feat = self.region_agg(region_feat)
                region_feats.append(region_feat)
        
        region_feats = torch.stack(region_feats, dim=1)  # [B, N_regions, hidden]
        
        # Build region-level graph (fully connected for simplicity)
        A_region = torch.ones(B, self.n_regions, self.n_regions, device=device)
        
        # Region-level processing
        final_out = self.region_gnn(region_feats, A_region, mc_samples=mc_samples)
        
        # Add channel-level attention for interpretability
        final_out['channel_attention'] = channel_out['attention']
        
        return final_out