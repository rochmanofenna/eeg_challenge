"""
Data I/O utilities for BEF pipeline
Interfaces with eegdash and braindecode for data loading
"""

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset
from pathlib import Path
from typing import Tuple, Optional, Dict, List


class EEGDataset(Dataset):
    """
    EEG Dataset wrapper for challenge data
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        task: str = "challenge1",
        transform=None
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.task = task
        self.transform = transform
        
        # Load data (placeholder - replace with actual loading)
        self.data, self.targets = self._load_data()
        
    def _load_data(self):
        """Load EEG data from files"""
        # Placeholder implementation
        # In practice, this would load from .npy, .mat, or .fif files
        n_samples = 1000 if self.split == "train" else 200
        data = np.random.randn(n_samples, 129, 200).astype(np.float32)
        
        if self.task == "challenge1":
            # Reaction time regression
            targets = 200 + np.random.randn(n_samples) * 50
        else:
            # Psychopathology factors
            targets = np.random.randn(n_samples, 4)
        
        return torch.tensor(data), torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


def load_eeg_data(
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    task: str = "challenge1"
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Load EEG data for training
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Create datasets
    train_dataset = EEGDataset(data_path, split="train", task=task)
    val_dataset = EEGDataset(data_path, split="val", task=task)
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Optional test loader
    test_loader = None
    if Path(data_path, "test").exists():
        test_dataset = EEGDataset(data_path, split="test", task=task)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
    
    return train_loader, val_loader, test_loader


class EEGAugmentation:
    """
    Data augmentation for EEG signals
    """
    
    def __init__(
        self,
        noise_level: float = 0.1,
        amplitude_scale: Tuple[float, float] = (0.9, 1.1),
        temporal_shift: int = 5
    ):
        self.noise_level = noise_level
        self.amplitude_scale = amplitude_scale
        self.temporal_shift = temporal_shift
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentations
        
        Args:
            x: EEG data [C, T]
        """
        # Add noise
        if self.noise_level > 0:
            x = x + torch.randn_like(x) * self.noise_level
        
        # Scale amplitude
        scale = np.random.uniform(*self.amplitude_scale)
        x = x * scale
        
        # Temporal shift
        if self.temporal_shift > 0:
            shift = np.random.randint(-self.temporal_shift, self.temporal_shift)
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=-1)
        
        return x


def preprocess_eeg(
    data: np.ndarray,
    sfreq: int = 100,
    lowpass: float = 40,
    highpass: float = 0.5,
    notch: Optional[float] = 50
) -> np.ndarray:
    """
    Preprocess EEG data
    
    Args:
        data: Raw EEG [n_samples, n_channels, n_times]
        sfreq: Sampling frequency
        lowpass: Low-pass filter cutoff
        highpass: High-pass filter cutoff
        notch: Notch filter frequency (e.g., 50 or 60 Hz)
    
    Returns:
        Preprocessed EEG data
    """
    # Placeholder for preprocessing
    # In practice, would use MNE or scipy for filtering
    
    # Normalize
    mean = data.mean(axis=-1, keepdims=True)
    std = data.std(axis=-1, keepdims=True)
    data = (data - mean) / (std + 1e-8)
    
    return data


def create_windows(
    data: np.ndarray,
    window_size: int = 200,
    stride: int = 100,
    drop_last: bool = True
) -> np.ndarray:
    """
    Create sliding windows from continuous EEG
    
    Args:
        data: EEG data [n_channels, n_times]
        window_size: Window length in samples
        stride: Step size between windows
        drop_last: Drop incomplete window at end
    
    Returns:
        Windows [n_windows, n_channels, window_size]
    """
    n_channels, n_times = data.shape
    windows = []
    
    for start in range(0, n_times - window_size + 1, stride):
        end = start + window_size
        if end <= n_times:
            windows.append(data[:, start:end])
        elif not drop_last:
            # Pad last window
            window = np.zeros((n_channels, window_size))
            window[:, :n_times-start] = data[:, start:]
            windows.append(window)
    
    return np.array(windows)


def load_braindecode_model(model_name: str = "EEGNet"):
    """
    Load a pretrained Braindecode model
    
    Args:
        model_name: Name of the model to load
    
    Returns:
        Pretrained model (if available)
    """
    # Placeholder - would interface with Braindecode
    print(f"Would load Braindecode model: {model_name}")
    return None


def save_predictions(
    predictions: np.ndarray,
    output_path: str,
    participant_ids: Optional[List[str]] = None
):
    """
    Save predictions in challenge format
    
    Args:
        predictions: Model predictions
        output_path: Where to save
        participant_ids: Optional participant IDs
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy array
    np.save(output_path, predictions)
    
    # Also save as CSV if IDs provided
    if participant_ids:
        import pandas as pd
        df = pd.DataFrame({
            'participant_id': participant_ids,
            'prediction': predictions.flatten()
        })
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)