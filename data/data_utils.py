"""
Data utilities for EEG Challenge
Handles data loading, preprocessing, and augmentation
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, List
import pandas as pd

# Optional imports for when we have the full EEG pipeline
try:
    import mne
    from braindecode.preprocessing import preprocess, Preprocessor
    from braindecode.datasets import create_windows_from_events
    HAS_EEG_TOOLS = True
except ImportError:
    HAS_EEG_TOOLS = False
    print("Warning: EEG tools not available, using mock implementations")


class EEGAugmentation:
    """EEG-specific data augmentations"""
    
    def __init__(
        self,
        channel_drop_prob: float = 0.1,
        time_mask_prob: float = 0.1,
        gaussian_noise_std: float = 0.1,
        time_shift_samples: int = 10
    ):
        self.channel_drop_prob = channel_drop_prob
        self.time_mask_prob = time_mask_prob
        self.gaussian_noise_std = gaussian_noise_std
        self.time_shift_samples = time_shift_samples
        
    def __call__(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Apply augmentations to EEG data
        
        Args:
            x: EEG data [batch, channels, time] or [channels, time]
            training: Whether in training mode
            
        Returns:
            Augmented EEG data
        """
        if not training:
            return x
            
        x = x.clone()
        
        # Channel dropout (simulate broken electrodes)
        if self.channel_drop_prob > 0 and torch.rand(1).item() < 0.5:
            n_channels = x.shape[-2]
            n_drop = int(n_channels * self.channel_drop_prob)
            if n_drop > 0:
                drop_idx = torch.randperm(n_channels)[:n_drop]
                x[..., drop_idx, :] = 0
                
        # Time masking (simulate artifacts)
        if self.time_mask_prob > 0 and torch.rand(1).item() < 0.3:
            n_times = x.shape[-1]
            mask_len = int(n_times * self.time_mask_prob)
            if mask_len > 0:
                start = torch.randint(0, n_times - mask_len, (1,)).item()
                x[..., start:start + mask_len] = 0
                
        # Gaussian noise
        if self.gaussian_noise_std > 0:
            noise = torch.randn_like(x) * self.gaussian_noise_std
            x = x + noise
            
        # Time shift (circular)
        if self.time_shift_samples > 0 and torch.rand(1).item() < 0.3:
            shift = torch.randint(-self.time_shift_samples, self.time_shift_samples, (1,)).item()
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=-1)
                
        return x


class SubjectNormalization:
    """Per-subject normalization to handle inter-subject variability"""
    
    def __init__(self, method: str = "robust"):
        self.method = method
        self.stats = {}
        
    def fit(self, X: np.ndarray, subject_ids: np.ndarray):
        """Compute normalization statistics per subject"""
        unique_subjects = np.unique(subject_ids)
        
        for subj in unique_subjects:
            mask = subject_ids == subj
            subj_data = X[mask]
            
            if self.method == "robust":
                # Robust scaling using median and MAD
                median = np.median(subj_data, axis=(0, 2), keepdims=True)
                mad = np.median(np.abs(subj_data - median), axis=(0, 2), keepdims=True)
                self.stats[subj] = {"center": median, "scale": mad + 1e-8}
            else:
                # Standard scaling
                mean = np.mean(subj_data, axis=(0, 2), keepdims=True)
                std = np.std(subj_data, axis=(0, 2), keepdims=True)
                self.stats[subj] = {"center": mean, "scale": std + 1e-8}
                
    def transform(self, X: np.ndarray, subject_ids: np.ndarray) -> np.ndarray:
        """Apply normalization"""
        X_norm = np.zeros_like(X)
        
        for subj in np.unique(subject_ids):
            mask = subject_ids == subj
            if subj in self.stats:
                stats = self.stats[subj]
                X_norm[mask] = (X[mask] - stats["center"]) / stats["scale"]
            else:
                # Fallback to global normalization for unseen subjects
                X_norm[mask] = (X[mask] - np.mean(X[mask])) / (np.std(X[mask]) + 1e-8)
                
        return X_norm


class EEGChallengeDataset(Dataset):
    """
    Custom dataset for EEG Challenge with proper preprocessing
    """
    
    def __init__(
        self,
        windows_dataset,
        transform: Optional[EEGAugmentation] = None,
        normalizer: Optional[SubjectNormalization] = None,
        target_type: str = "regression",  # "regression" or "classification"
        cache_data: bool = True
    ):
        self.windows_dataset = windows_dataset
        self.transform = transform
        self.normalizer = normalizer
        self.target_type = target_type
        self.cache_data = cache_data
        
        # Extract metadata
        self.metadata = windows_dataset.get_metadata()
        
        # Cache preprocessed data if requested
        if cache_data:
            self._cache = {}
            
    def __len__(self):
        return len(self.windows_dataset)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        if self.cache_data and idx in self._cache:
            X, y, info = self._cache[idx]
        else:
            # Get raw window
            X, y = self.windows_dataset[idx]
            
            # Convert to tensor if needed
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()
            if isinstance(y, np.ndarray):
                y = torch.from_numpy(y).float()
                
            # Get metadata
            info = self.metadata.iloc[idx].to_dict()
            
            # Apply normalization if available
            if self.normalizer is not None and "subject" in info:
                subject_id = info["subject"]
                X_np = X.numpy()
                X_np = self.normalizer.transform(
                    X_np[np.newaxis, ...], 
                    np.array([subject_id])
                )[0]
                X = torch.from_numpy(X_np).float()
                
            if self.cache_data:
                self._cache[idx] = (X.clone(), y.clone(), info.copy())
                
        # Apply augmentations
        if self.transform is not None:
            X = self.transform(X, training=self.training)
            
        # Handle target based on task type
        if self.target_type == "classification" and y.dim() > 0:
            # Convert regression target to classification if needed
            # This is task-specific logic
            pass
            
        return X, y, info
        
    def train(self):
        self.training = True
        return self
        
    def eval(self):
        self.training = False
        return self


def create_data_loaders(
    dataset_ccd,
    dataset_sus,
    batch_size: int = 32,
    num_workers: int = 4,
    valid_split: float = 0.1,
    test_split: float = 0.1,
    augmentation_config: Optional[dict] = None
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and testing
    
    Returns:
        Dictionary with train/valid/test loaders for both tasks
    """
    # Default augmentation config
    if augmentation_config is None:
        augmentation_config = {
            "channel_drop_prob": 0.1,
            "time_mask_prob": 0.05,
            "gaussian_noise_std": 0.05,
            "time_shift_samples": 5
        }
        
    # Create augmentation transform
    transform = EEGAugmentation(**augmentation_config)
    
    # Get metadata for splitting
    meta_ccd = dataset_ccd.get_metadata()
    meta_sus = dataset_sus.get_metadata()
    
    # Subject-based splitting to ensure no data leakage
    subjects_ccd = meta_ccd["subject"].unique()
    subjects_sus = meta_sus["subject"].unique()
    
    # Split subjects
    n_valid_ccd = int(len(subjects_ccd) * valid_split)
    n_test_ccd = int(len(subjects_ccd) * test_split)
    n_valid_sus = int(len(subjects_sus) * valid_split)
    n_test_sus = int(len(subjects_sus) * test_split)
    
    # Random shuffle subjects
    np.random.seed(42)
    subjects_ccd = np.random.permutation(subjects_ccd)
    subjects_sus = np.random.permutation(subjects_sus)
    
    # Create splits
    test_subjects_ccd = subjects_ccd[:n_test_ccd]
    valid_subjects_ccd = subjects_ccd[n_test_ccd:n_test_ccd + n_valid_ccd]
    train_subjects_ccd = subjects_ccd[n_test_ccd + n_valid_ccd:]
    
    test_subjects_sus = subjects_sus[:n_test_sus]
    valid_subjects_sus = subjects_sus[n_test_sus:n_test_sus + n_valid_sus]
    train_subjects_sus = subjects_sus[n_test_sus + n_valid_sus:]
    
    # Create index masks
    train_idx_ccd = meta_ccd["subject"].isin(train_subjects_ccd)
    valid_idx_ccd = meta_ccd["subject"].isin(valid_subjects_ccd)
    test_idx_ccd = meta_ccd["subject"].isin(test_subjects_ccd)
    
    train_idx_sus = meta_sus["subject"].isin(train_subjects_sus)
    valid_idx_sus = meta_sus["subject"].isin(valid_subjects_sus)
    test_idx_sus = meta_sus["subject"].isin(test_subjects_sus)
    
    # Create normalizers
    normalizer_ccd = SubjectNormalization(method="robust")
    normalizer_sus = SubjectNormalization(method="robust")
    
    # Fit normalizers on training data only
    # This would require loading the actual data - simplified here
    
    # Create datasets
    datasets = {
        "ccd_train": EEGChallengeDataset(
            dataset_ccd[train_idx_ccd],
            transform=transform,
            normalizer=normalizer_ccd,
            target_type="regression"
        ),
        "ccd_valid": EEGChallengeDataset(
            dataset_ccd[valid_idx_ccd],
            transform=None,  # No augmentation for validation
            normalizer=normalizer_ccd,
            target_type="regression"
        ),
        "ccd_test": EEGChallengeDataset(
            dataset_ccd[test_idx_ccd],
            transform=None,
            normalizer=normalizer_ccd,
            target_type="regression"
        ),
        "sus_train": EEGChallengeDataset(
            dataset_sus[train_idx_sus],
            transform=transform,
            normalizer=normalizer_sus,
            target_type="regression"
        ),
        "sus_valid": EEGChallengeDataset(
            dataset_sus[valid_idx_sus],
            transform=None,
            normalizer=normalizer_sus,
            target_type="regression"
        ),
        "sus_test": EEGChallengeDataset(
            dataset_sus[test_idx_sus],
            transform=None,
            normalizer=normalizer_sus,
            target_type="regression"
        ),
    }
    
    # Create data loaders
    loaders = {}
    for name, dataset in datasets.items():
        shuffle = "train" in name
        dataset.train() if "train" in name else dataset.eval()
        
        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=shuffle  # Drop last batch for training to avoid batch norm issues
        )
        
    return loaders


class EEGBatchSampler:
    """
    Custom batch sampler that ensures balanced batches
    across subjects and conditions
    """
    
    def __init__(
        self,
        metadata: pd.DataFrame,
        batch_size: int,
        balance_by: List[str] = ["subject", "correct"]
    ):
        self.metadata = metadata
        self.batch_size = batch_size
        self.balance_by = balance_by
        
        # Group indices by balancing criteria
        self.groups = {}
        for _, group in metadata.groupby(balance_by):
            key = tuple(group.iloc[0][balance_by])
            self.groups[key] = group.index.tolist()
            
    def __iter__(self):
        # Shuffle indices within each group
        for indices in self.groups.values():
            np.random.shuffle(indices)
            
        # Create balanced batches
        batch = []
        group_iters = {k: iter(v) for k, v in self.groups.items()}
        
        while group_iters:
            # Sample from each group
            for key in list(group_iters.keys()):
                try:
                    idx = next(group_iters[key])
                    batch.append(idx)
                    
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
                except StopIteration:
                    del group_iters[key]
                    
        # Yield remaining samples
        if batch:
            yield batch
            
    def __len__(self):
        return len(self.metadata) // self.batch_size