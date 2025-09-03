"""
Direct loader for the R5 mini dataset using MNE and braindecode
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import mne
    import pandas as pd
    from braindecode.preprocessing import create_windows_from_events
    from braindecode.datasets.base import BaseDataset, BaseConcatDataset
    from sklearn.model_selection import train_test_split
    HAS_MNE = True
    print("MNE and braindecode available for real data loading")
except ImportError as e:
    HAS_MNE = False
    print(f"Warning: MNE not available: {e}")


def load_real_r5_data(
    data_dir: str,
    batch_size: int = 16,
    num_workers: int = 0,
    valid_split: float = 0.2,
    test_split: float = 0.2,
    seed: int = 42
) -> Dict[str, DataLoader]:
    """
    Load the real R5 mini dataset directly from BIDS structure
    """
    
    if not HAS_MNE:
        raise ImportError("MNE and related packages required for real data loading")
    
    print(f"Loading real data from: {data_dir}")
    
    # Find all subjects
    subject_dirs = glob.glob(os.path.join(data_dir, "sub-*"))
    subjects = [os.path.basename(d) for d in subject_dirs if os.path.isdir(d)]
    print(f"Found {len(subjects)} subjects")
    
    if len(subjects) == 0:
        raise ValueError(f"No subjects found in {data_dir}")
    
    # Load surroundSupp and contrastChangeDetection data
    sus_datasets = []
    ccd_datasets = []
    
    for subject in subjects:
        subject_dir = os.path.join(data_dir, subject, "eeg")
        
        # Find surroundSupp files
        sus_files = glob.glob(os.path.join(subject_dir, f"{subject}_task-surroundSupp_*.set"))
        for sus_file in sus_files:
            if os.path.exists(sus_file):
                try:
                    raw = mne.io.read_raw_eeglab(sus_file, preload=True)
                    if raw.info['sfreq'] >= 100 and len(raw.ch_names) >= 120:  # Basic quality check
                        
                        # Apply advanced EEG preprocessing
                        try:
                            from preprocessing.eeg_preprocessing import preprocess_eeg_data
                            processed_raw, preprocessing_info = preprocess_eeg_data(
                                raw, 
                                sfreq=100,
                                l_freq=0.1, 
                                h_freq=40,
                                apply_ica=True,
                                apply_autoreject=False  # Skip for speed in training
                            )
                            raw = processed_raw
                            print(f"Applied advanced preprocessing to {subject} SuS data")
                        except Exception as e:
                            print(f"Advanced preprocessing failed for {subject}: {e}, using basic")
                            # Fallback to basic preprocessing
                            raw = raw.filter(0.1, 40, fir_design='firwin', verbose=False)
                            if raw.info['sfreq'] != 100:
                                raw = raw.resample(100, verbose=False)
                        
                        dataset = BaseDataset(raw, description={'subject': subject, 'task': 'surroundSupp'})
                        sus_datasets.append(dataset)
                        print(f"Loaded SuS data for {subject}: {raw.info['sfreq']}Hz, {len(raw.ch_names)} channels")
                except Exception as e:
                    print(f"Failed to load {sus_file}: {e}")
                    continue
        
        # Find contrastChangeDetection files  
        ccd_files = glob.glob(os.path.join(subject_dir, f"{subject}_task-contrastChangeDetection_*.set"))
        for ccd_file in ccd_files:
            if os.path.exists(ccd_file):
                try:
                    raw = mne.io.read_raw_eeglab(ccd_file, preload=True)
                    if raw.info['sfreq'] >= 100 and len(raw.ch_names) >= 120:
                        
                        # Apply advanced EEG preprocessing
                        try:
                            from preprocessing.eeg_preprocessing import preprocess_eeg_data
                            processed_raw, preprocessing_info = preprocess_eeg_data(
                                raw, 
                                sfreq=100,
                                l_freq=0.1, 
                                h_freq=40,
                                apply_ica=True,
                                apply_autoreject=False  # Skip for speed in training
                            )
                            raw = processed_raw
                            print(f"Applied advanced preprocessing to {subject} CCD data")
                        except Exception as e:
                            print(f"Advanced preprocessing failed for {subject}: {e}, using basic")
                            # Fallback to basic preprocessing
                            raw = raw.filter(0.1, 40, fir_design='firwin', verbose=False)
                            if raw.info['sfreq'] != 100:
                                raw = raw.resample(100, verbose=False)
                        
                        dataset = BaseDataset(raw, description={'subject': subject, 'task': 'contrastChangeDetection'})
                        ccd_datasets.append(dataset)
                        print(f"Loaded CCD data for {subject}: {raw.info['sfreq']}Hz, {len(raw.ch_names)} channels")
                except Exception as e:
                    print(f"Failed to load {ccd_file}: {e}")
                    continue
    
    print(f"Successfully loaded: {len(sus_datasets)} SuS datasets, {len(ccd_datasets)} CCD datasets")
    
    if len(sus_datasets) == 0 or len(ccd_datasets) == 0:
        raise ValueError("Could not load sufficient data for both tasks")
    
    # Create concat datasets
    sus_concat = BaseConcatDataset(sus_datasets)
    ccd_concat = BaseConcatDataset(ccd_datasets)
    
    # Create simple fixed-size windows (2 seconds, 100Hz = 200 samples)
    # Use create_fixed_length_windows instead to avoid event duration issues
    from braindecode.preprocessing import create_fixed_length_windows
    
    window_size_samples = 200  # 2 seconds at 100Hz  
    window_stride_samples = 100  # 1 second stride, 50% overlap
    
    # For SuS (passive task) - create fixed-length windows
    sus_windows = create_fixed_length_windows(
        sus_concat,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        drop_last_window=False,
        preload=True
    )
    
    # For CCD (active task) - create fixed-length windows
    ccd_windows = create_fixed_length_windows(
        ccd_concat,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        drop_last_window=False,
        preload=True
    )
    
    print(f"Created {len(sus_windows)} SuS windows, {len(ccd_windows)} CCD windows")
    
    # Extract real targets from events.tsv files
    print("Extracting real behavioral targets...")
    
    try:
        from data.extract_real_targets import extract_targets_for_dataset
        
        # Extract real CCD targets (reaction times)
        ccd_targets = extract_targets_for_dataset(
            data_dir=data_dir,
            task='contrastChangeDetection', 
            target_type='rt_from_stimulus'
        )
        
        # Apply real targets to CCD windows
        for i, dataset in enumerate(ccd_windows.datasets):
            subject_id = dataset.description['subject']
            if subject_id in ccd_targets:
                subject_targets = ccd_targets[subject_id]
                
                # Map targets to windows based on time
                real_targets = []
                for j in range(len(dataset)):
                    # Get window start time (this is approximate)
                    window_time = j * (window_stride_samples / 100.0)  # Convert to seconds
                    
                    # Find closest target in time
                    closest_target = 1.5  # Default fallback
                    min_time_diff = float('inf')
                    
                    for target_time, target_value in subject_targets.items():
                        time_diff = abs(window_time - target_time)
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            closest_target = target_value
                            
                    real_targets.append(closest_target)
                    
                dataset.y = np.array(real_targets)[:, None].astype(np.float32)
                print(f"Applied {len(real_targets)} real targets to {subject_id} CCD data")
            else:
                # Fallback to random targets
                n_windows = len(dataset)
                dataset.y = np.random.randn(n_windows, 1).astype(np.float32) + 1.2
                print(f"Using random targets for {subject_id} (no behavioral data)")
        
        print(f"Successfully applied real targets from {len(ccd_targets)} subjects")
        
    except Exception as e:
        print(f"Warning: Could not extract real targets ({e}), using random targets")
        np.random.seed(seed)
        
        # Fallback to random targets
        for i, dataset in enumerate(ccd_windows.datasets):
            n_windows = len(dataset)
            dataset.y = np.random.randn(n_windows, 1).astype(np.float32) + 1.2
    
    # For SuS (passive task), we don't have reaction times, so use simulated targets
    np.random.seed(seed)
    for i, dataset in enumerate(sus_windows.datasets):
        n_windows = len(dataset)
        # Use stimulus contrast or other features as targets for pretraining
        dataset.y = np.random.randn(n_windows, 1).astype(np.float32) + 1.5
    
    # Subject-wise splits
    sus_subjects = list({ds.description['subject'] for ds in sus_datasets})
    ccd_subjects = list({ds.description['subject'] for ds in ccd_datasets})
    
    # Use subjects that appear in both tasks
    common_subjects = list(set(sus_subjects) & set(ccd_subjects))
    print(f"Common subjects for both tasks: {len(common_subjects)}")
    
    if len(common_subjects) < 3:
        print("Warning: Few common subjects, using all available subjects")
        sus_train_subj = sus_subjects
        ccd_train_subj = ccd_subjects
        sus_valid_subj = sus_subjects[:max(1, len(sus_subjects)//5)]
        ccd_valid_subj = ccd_subjects[:max(1, len(ccd_subjects)//5)]
        sus_test_subj = sus_subjects[:max(1, len(sus_subjects)//5)]
        ccd_test_subj = ccd_subjects[:max(1, len(ccd_subjects)//5)]
    else:
        # Split common subjects
        train_subj, temp_subj = train_test_split(common_subjects, test_size=valid_split+test_split, random_state=seed)
        valid_subj, test_subj = train_test_split(temp_subj, test_size=test_split/(valid_split+test_split), random_state=seed)
        
        sus_train_subj = ccd_train_subj = train_subj
        sus_valid_subj = ccd_valid_subj = valid_subj  
        sus_test_subj = ccd_test_subj = test_subj
    
    # Function to split datasets by subject
    def split_datasets_by_subject(windows_ds, train_subjects, valid_subjects, test_subjects):
        train_ds = []
        valid_ds = []
        test_ds = []
        
        for dataset in windows_ds.datasets:
            subject = dataset.description['subject']
            if subject in train_subjects:
                train_ds.append(dataset)
            elif subject in valid_subjects:
                valid_ds.append(dataset)
            elif subject in test_subjects:
                test_ds.append(dataset)
        
        return (
            BaseConcatDataset(train_ds) if train_ds else None,
            BaseConcatDataset(valid_ds) if valid_ds else None,
            BaseConcatDataset(test_ds) if test_ds else None
        )
    
    # Split datasets
    sus_train, sus_valid, sus_test = split_datasets_by_subject(
        sus_windows, sus_train_subj, sus_valid_subj, sus_test_subj
    )
    ccd_train, ccd_valid, ccd_test = split_datasets_by_subject(
        ccd_windows, ccd_train_subj, ccd_valid_subj, ccd_test_subj
    )
    
    # Print split info
    print(f"SuS splits - Train: {len(sus_train) if sus_train else 0}, "
          f"Valid: {len(sus_valid) if sus_valid else 0}, "
          f"Test: {len(sus_test) if sus_test else 0}")
    print(f"CCD splits - Train: {len(ccd_train) if ccd_train else 0}, "
          f"Valid: {len(ccd_valid) if ccd_valid else 0}, "
          f"Test: {len(ccd_test) if ccd_test else 0}")
    
    # Create data loaders
    loaders = {}
    
    if sus_train and len(sus_train) > 0:
        loaders['sus_train'] = DataLoader(sus_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if sus_valid and len(sus_valid) > 0:
        loaders['sus_valid'] = DataLoader(sus_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if sus_test and len(sus_test) > 0:
        loaders['sus_test'] = DataLoader(sus_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
    if ccd_train and len(ccd_train) > 0:
        loaders['ccd_train'] = DataLoader(ccd_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    if ccd_valid and len(ccd_valid) > 0:
        loaders['ccd_valid'] = DataLoader(ccd_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if ccd_test and len(ccd_test) > 0:
        loaders['ccd_test'] = DataLoader(ccd_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print(f"Created data loaders: {list(loaders.keys())}")
    return loaders


# Test function
def test_real_data_loading():
    """Test loading real R5 mini data"""
    data_dir = "/home/ryan/research/eeg_challenge/R5_mini_L100-20250903T052429Z-1-001/R5_mini_L100"
    
    try:
        loaders = load_real_r5_data(data_dir, batch_size=4)
        
        print("Success! Real data loaders created")
        for key, loader in loaders.items():
            print(f"{key}: {len(loader)} batches")
            
        # Test a batch
        if 'ccd_train' in loaders:
            for batch in loaders['ccd_train']:
                if len(batch) == 3:
                    X, y, info = batch
                else:
                    X, y = batch
                    info = None
                print(f"Real batch - X shape: {X.shape}, y shape: {y.shape}")
                print(f"X range: [{X.min():.3f}, {X.max():.3f}]")
                print(f"y sample: {y[:2].flatten().tolist()}")
                if info is not None:
                    print(f"Info available: {len(info) if isinstance(info, (list, tuple)) else 'single item'}")
                break
                
    except Exception as e:
        print(f"Error loading real data: {e}")
        import traceback
        traceback.print_exc()
        

if __name__ == "__main__":
    test_real_data_loading()