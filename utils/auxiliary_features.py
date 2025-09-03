"""
Auxiliary feature pipeline using BICEP → ENN → FusionAlpha
Generates teacher signals and stability priors for EEG model training
"""

import os
import subprocess
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors


class BICEPFeatureGenerator:
    """
    Use BICEP to generate synthetic trajectories that match EEG statistics
    """
    
    def __init__(self, bicep_path: str = "../BICEPsrc/BICEPrust/bicep"):
        self.bicep_path = Path(bicep_path)
        if not self.bicep_path.exists():
            print(f"Warning: BICEP path {bicep_path} not found")
            
    def generate_trajectories_from_eeg(
        self,
        eeg_features: np.ndarray,
        output_path: str,
        n_trajectories: int = 1000,
        trajectory_length: int = 200
    ) -> Optional[pd.DataFrame]:
        """
        Generate BICEP trajectories that match EEG feature statistics
        
        Args:
            eeg_features: EEG features [n_samples, feature_dim]
            output_path: Path to save trajectories
            n_trajectories: Number of trajectories to generate
            trajectory_length: Length of each trajectory
            
        Returns:
            DataFrame with generated trajectories
        """
        # Compute statistics from EEG features
        mean = np.mean(eeg_features, axis=0)
        cov = np.cov(eeg_features.T)
        
        # Create BICEP config that matches these statistics
        config = {
            "dimension": len(mean),
            "drift": mean.tolist(),
            "diffusion": np.sqrt(np.diag(cov)).tolist(),
            "n_paths": n_trajectories,
            "n_steps": trajectory_length,
            "dt": 0.01,  # 10ms timestep (matching 100Hz EEG)
            "output": output_path
        }
        
        # Save config
        config_path = Path(output_path).parent / "bicep_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f)
            
        # Run BICEP
        if self.bicep_path.exists():
            cmd = [
                str(self.bicep_path / "target/release/bicep"),
                "--config", str(config_path),
                "--mode", "sde",  # Stochastic differential equation mode
                "--out", output_path
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    # Load generated trajectories
                    return pd.read_parquet(output_path)
                else:
                    print(f"BICEP error: {result.stderr}")
            except Exception as e:
                print(f"Error running BICEP: {e}")
                
        # Fallback: generate mock trajectories
        print("Using mock trajectory generation")
        trajectories = []
        for i in range(n_trajectories):
            # Random walk with drift matching EEG statistics
            traj = np.zeros((trajectory_length, len(mean)))
            traj[0] = np.random.multivariate_normal(mean, cov)
            
            for t in range(1, trajectory_length):
                drift = 0.99 * traj[t-1] + 0.01 * mean
                noise = np.random.multivariate_normal(np.zeros_like(mean), 0.01 * cov)
                traj[t] = drift + noise
                
            trajectories.append(traj.flatten())
            
        df = pd.DataFrame(trajectories)
        df.to_parquet(output_path)
        return df


class ENNPriorComputer:
    """
    Use ENN C++ to compute committor functions and uncertainty estimates
    """
    
    def __init__(self, enn_path: str = "../enn-cpp"):
        self.enn_path = Path(enn_path)
        if not self.enn_path.exists():
            print(f"Warning: ENN path {enn_path} not found")
            
    def prepare_enn_input(
        self,
        trajectories_df: pd.DataFrame,
        labels: Optional[np.ndarray] = None
    ) -> str:
        """
        Convert trajectories to ENN input format
        """
        # ENN expects CSV with columns: t, s0, s1, ..., sD, target
        n_samples = len(trajectories_df)
        n_features = trajectories_df.shape[1]
        
        # Reshape if needed
        if n_features > 200:  # Flattened trajectories
            n_timesteps = 200
            n_dims = n_features // n_timesteps
            
            # Create time-indexed format
            rows = []
            for i in range(n_samples):
                traj = trajectories_df.iloc[i].values.reshape(n_timesteps, n_dims)
                for t in range(n_timesteps):
                    row = [t] + traj[t].tolist()
                    if labels is not None:
                        row.append(labels[i])
                    else:
                        row.append(0.5)  # Default target
                    rows.append(row)
                    
            # Create DataFrame
            columns = ['t'] + [f's{i}' for i in range(n_dims)] + ['target']
            enn_df = pd.DataFrame(rows, columns=columns)
        else:
            # Already in correct format
            enn_df = trajectories_df.copy()
            if labels is not None:
                enn_df['target'] = np.repeat(labels, len(trajectories_df) // len(labels))
                
        # Save to CSV
        csv_path = self.enn_path / "data" / "eeg_trajectories.csv"
        csv_path.parent.mkdir(exist_ok=True)
        enn_df.to_csv(csv_path, index=False)
        
        return str(csv_path)
        
    def compute_priors(
        self,
        input_csv: str,
        output_csv: str,
        epochs: int = 100,
        lr: float = 1e-3
    ) -> Optional[np.ndarray]:
        """
        Run ENN to compute committor priors
        
        Returns:
            Array of prior probabilities
        """
        if self.enn_path.exists():
            # Build command
            cmd = [
                str(self.enn_path / "apps" / "committor_train"),
                "--train", input_csv,
                "--epochs", str(epochs),
                "--lr", str(lr),
                "--out", output_csv
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    # Load predictions
                    pred_df = pd.read_csv(output_csv)
                    return pred_df['enn_prior'].values
                else:
                    print(f"ENN error: {result.stderr}")
            except Exception as e:
                print(f"Error running ENN: {e}")
                
        # Fallback: compute mock priors
        print("Using mock prior computation")
        input_df = pd.read_csv(input_csv)
        
        # Simple committor based on distance to target
        if 'target' in input_df.columns:
            targets = input_df['target'].values
            # Smooth the targets to create priors
            from scipy.ndimage import gaussian_filter1d
            priors = gaussian_filter1d(targets, sigma=10)
        else:
            # Random priors
            priors = np.random.beta(2, 2, size=len(input_df))
            
        return priors


class FusionAlphaRefiner:
    """
    Use FusionAlpha to refine priors using subject/trial relationships
    """
    
    def __init__(self, use_python_bindings: bool = True):
        self.use_python_bindings = use_python_bindings
        
        if use_python_bindings:
            try:
                import fusion_bindings as fb
                self.fb = fb
                self.has_bindings = True
            except ImportError:
                print("Warning: fusion_bindings not found, using Python implementation")
                self.has_bindings = False
        else:
            self.has_bindings = False
            
    def build_subject_graph(
        self,
        metadata: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        k_neighbors: int = 10
    ) -> List[Tuple[int, int]]:
        """
        Build k-NN graph based on subject similarity
        
        Args:
            metadata: DataFrame with subject information
            feature_cols: Columns to use for similarity (age, scores, etc.)
            k_neighbors: Number of neighbors
            
        Returns:
            List of edges (i, j)
        """
        if feature_cols is None:
            # Default: use any numeric columns
            feature_cols = metadata.select_dtypes(include=[np.number]).columns.tolist()
            
        if not feature_cols:
            # No features, create random graph
            n = len(metadata)
            edges = []
            for i in range(n):
                neighbors = np.random.choice(n, size=min(k_neighbors, n-1), replace=False)
                neighbors = neighbors[neighbors != i]
                for j in neighbors:
                    edges.append((i, j))
            return edges
            
        # Extract features
        features = metadata[feature_cols].fillna(0).values
        
        # Normalize features
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        # Build k-NN graph
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1)
        nbrs.fit(features)
        
        edges = []
        for i in range(len(features)):
            distances, indices = nbrs.kneighbors([features[i]])
            # Skip self (first neighbor)
            for j in indices[0][1:]:
                edges.append((i, int(j)))
                
        return edges
        
    def refine_priors(
        self,
        priors: np.ndarray,
        edges: List[Tuple[int, int]],
        beta: float = 3.0,
        iterations: int = 10
    ) -> np.ndarray:
        """
        Refine priors using graph propagation
        
        Args:
            priors: Initial prior probabilities
            edges: Graph edges
            beta: Propagation strength
            iterations: Number of iterations
            
        Returns:
            Refined posteriors
        """
        if self.has_bindings:
            # Use Rust implementation
            nodes = list(range(len(priors)))
            posteriors = self.fb.simple_propagate(
                nodes, edges, priors.tolist(), beta, iterations
            )
            return np.array(posteriors)
        else:
            # Python implementation
            return self._python_propagate(priors, edges, beta, iterations)
            
    def _python_propagate(
        self,
        priors: np.ndarray,
        edges: List[Tuple[int, int]],
        beta: float,
        iterations: int
    ) -> np.ndarray:
        """
        Simple Python implementation of belief propagation
        """
        n = len(priors)
        beliefs = priors.copy()
        
        # Build adjacency list
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
            
        # Iterate
        for _ in range(iterations):
            new_beliefs = beliefs.copy()
            
            for i in range(n):
                if not adj[i]:
                    continue
                    
                # Aggregate neighbor beliefs
                neighbor_beliefs = [beliefs[j] for j in adj[i]]
                avg_neighbor = np.mean(neighbor_beliefs)
                
                # Update with weighted average
                alpha = 1.0 / (1.0 + beta)
                new_beliefs[i] = alpha * priors[i] + (1 - alpha) * avg_neighbor
                
            beliefs = new_beliefs
            
        return beliefs


class AuxiliaryFeaturePipeline:
    """
    Complete pipeline: EEG → BICEP → ENN → FusionAlpha → Features
    """
    
    def __init__(
        self,
        bicep_path: Optional[str] = None,
        enn_path: Optional[str] = None,
        cache_dir: str = "./aux_features_cache"
    ):
        self.bicep = BICEPFeatureGenerator(bicep_path) if bicep_path else None
        self.enn = ENNPriorComputer(enn_path) if enn_path else None
        self.fusion = FusionAlphaRefiner()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def generate_features(
        self,
        eeg_features: np.ndarray,
        metadata: pd.DataFrame,
        labels: Optional[np.ndarray] = None,
        use_cache: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Generate auxiliary features for EEG data
        
        Args:
            eeg_features: EEG features [n_samples, feature_dim]
            metadata: Metadata with subject info
            labels: Optional labels for supervised prior learning
            use_cache: Whether to use cached results
            
        Returns:
            Dictionary of auxiliary features
        """
        cache_key = f"features_{hash(eeg_features.tobytes())}"
        cache_path = self.cache_dir / f"{cache_key}.npz"
        
        if use_cache and cache_path.exists():
            print("Loading cached auxiliary features")
            return dict(np.load(cache_path))
            
        features = {}
        
        # Step 1: Generate BICEP trajectories
        print("Generating BICEP trajectories...")
        traj_path = self.cache_dir / "trajectories.parquet"
        
        if self.bicep:
            trajectories = self.bicep.generate_trajectories_from_eeg(
                eeg_features, str(traj_path)
            )
        else:
            # Mock trajectories
            trajectories = pd.DataFrame(
                np.random.randn(len(eeg_features), 200 * 8)  # 200 timesteps, 8 dims
            )
            
        # Step 2: Compute ENN priors
        print("Computing ENN priors...")
        if self.enn and trajectories is not None:
            input_csv = self.enn.prepare_enn_input(trajectories, labels)
            output_csv = str(self.cache_dir / "enn_predictions.csv")
            priors = self.enn.compute_priors(input_csv, output_csv)
        else:
            # Mock priors
            priors = np.random.beta(2, 2, size=len(eeg_features))
            
        features['enn_prior'] = priors
        
        # Step 3: Build subject graph and refine with FusionAlpha
        print("Refining with FusionAlpha...")
        edges = self.fusion.build_subject_graph(metadata, k_neighbors=10)
        posteriors = self.fusion.refine_priors(priors, edges, beta=3.0, iterations=10)
        
        features['fusion_posterior'] = posteriors
        
        # Step 4: Compute derived features
        features['uncertainty'] = np.abs(priors - posteriors)
        features['entropy'] = -priors * np.log(priors + 1e-8) - (1-priors) * np.log(1-priors + 1e-8)
        features['confidence'] = np.maximum(posteriors, 1 - posteriors)
        
        # Compute stability metric (how much the prior changed)
        features['stability'] = 1.0 - np.abs(priors - posteriors)
        
        # Save to cache
        if use_cache:
            np.savez(cache_path, **features)
            
        return features
        
    def create_feature_targets(
        self,
        features: Dict[str, np.ndarray],
        true_labels: np.ndarray,
        task: str = "regression"
    ) -> Dict[str, np.ndarray]:
        """
        Create auxiliary training targets from features
        
        These can be used for multi-task learning or distillation
        """
        targets = {}
        
        if task == "regression":
            # Use refined posteriors as soft targets
            targets['soft_target'] = features['fusion_posterior']
            
            # Create confidence-weighted targets
            confidence = features['confidence']
            targets['weighted_target'] = (
                confidence * true_labels + 
                (1 - confidence) * features['fusion_posterior']
            )
            
        elif task == "classification":
            # Create soft labels
            targets['soft_labels'] = features['fusion_posterior']
            
            # Temperature-scaled labels
            temperature = 3.0
            logits = np.log(features['fusion_posterior'] / (1 - features['fusion_posterior'] + 1e-8))
            targets['temp_labels'] = 1 / (1 + np.exp(-logits / temperature))
            
        # Uncertainty targets for auxiliary loss
        targets['uncertainty_target'] = features['uncertainty']
        
        return targets


def integrate_auxiliary_features(
    model: nn.Module,
    aux_features: Dict[str, np.ndarray],
    integration_type: str = "concatenate"
) -> nn.Module:
    """
    Integrate auxiliary features into the model
    
    Args:
        model: Base EEG model
        aux_features: Auxiliary features dictionary
        integration_type: How to integrate ("concatenate", "attention", "gating")
    """
    
    class AuxiliaryEnhancedModel(nn.Module):
        def __init__(self, base_model, aux_dim, integration_type):
            super().__init__()
            self.base_model = base_model
            self.aux_dim = aux_dim
            self.integration_type = integration_type
            
            if integration_type == "concatenate":
                # Simple concatenation
                old_head_in = base_model.head[0].in_features
                new_head_in = old_head_in + aux_dim
                
                # Replace first layer of head
                self.aux_projection = nn.Linear(aux_dim, aux_dim)
                self.new_head_first = nn.Linear(new_head_in, base_model.head[0].out_features)
                
            elif integration_type == "attention":
                # Attention-based fusion
                self.aux_attention = nn.MultiheadAttention(
                    embed_dim=64, num_heads=4, batch_first=True
                )
                
            elif integration_type == "gating":
                # Gating mechanism
                self.aux_gate = nn.Sequential(
                    nn.Linear(aux_dim, 64),
                    nn.Sigmoid()
                )
                
        def forward(self, x, aux_input=None):
            # Get base features
            if hasattr(self.base_model, 'feature_extractor'):
                features = self.base_model.feature_extractor(x)
                features = self.base_model.projection(features)
            else:
                features = self.base_model(x, return_features=True)[-1]
                
            if aux_input is not None:
                if self.integration_type == "concatenate":
                    aux_features = self.aux_projection(aux_input)
                    combined = torch.cat([features, aux_features], dim=1)
                    # Use new head
                    output = self.new_head_first(combined)
                    # Continue with rest of head
                    for layer in self.base_model.head[1:]:
                        output = layer(output)
                    return output
                    
                elif self.integration_type == "attention":
                    # Cross-attention between EEG and auxiliary features
                    attended, _ = self.aux_attention(
                        features.unsqueeze(1),
                        aux_input.unsqueeze(1),
                        aux_input.unsqueeze(1)
                    )
                    features = features + attended.squeeze(1)
                    
                elif self.integration_type == "gating":
                    # Gate EEG features with auxiliary information
                    gate = self.aux_gate(aux_input)
                    features = features * gate
                    
            # Continue with normal forward pass
            if hasattr(self.base_model, 'enn_cell'):
                return self.base_model.enn_cell(features)
            else:
                return self.base_model.head(features)
                
    # Get auxiliary dimension
    aux_dim = sum(v.shape[1] if v.ndim > 1 else 1 for v in aux_features.values())
    
    # Create enhanced model
    enhanced_model = AuxiliaryEnhancedModel(model, aux_dim, integration_type)
    
    return enhanced_model