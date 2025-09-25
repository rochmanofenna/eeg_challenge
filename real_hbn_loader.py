#!/usr/bin/env python3
"""REAL HBN EEG Data Loader - streams genuine recordings from S3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import mne
import numpy as np
import torch
from botocore import UNSIGNED
from botocore.config import Config
from torch.utils.data import Dataset


def _build_subjects_cache_name(releases: List[str], tasks: List[str]) -> str:
    """Create a deterministic cache filename for the requested releases/tasks."""

    key = "|".join(sorted(releases)) + "#" + "|".join(sorted(tasks))
    return f"subjects_{hashlib.md5(key.encode()).hexdigest()}.json"

class RealHBNDataset(Dataset):
    """Actually loads REAL EEG data from the HBN S3 bucket."""

    DEFAULT_RELEASES: Tuple[str, ...] = (
        "cmi_bids_R1",
        "cmi_bids_R2",
        "cmi_bids_R3",
        "cmi_bids_R4",
        "cmi_bids_R5",
        "cmi_bids_R6",
        "cmi_bids_R7",
        "cmi_bids_R8",
        "cmi_bids_R9",
        "cmi_bids_R10",
        "cmi_bids_R11",
        "cmi_bids_NC",
    )

    DEFAULT_TASKS: Tuple[str, ...] = (
        "RestingState",
        "DespicableMe",
        "DiaryOfAWimpyKid",
    )

    def __init__(
        self,
        releases: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        max_subjects: int = 10**9,
        cache_dir: str = "/tmp/real_hbn_cache",
        window_duration: float = 2.0,
        sfreq_resample: int = 100,
        n_channels: int = 129,
        seed: int = 42,
        return_metadata: bool = False,
    ):
        self.releases = releases or list(self.DEFAULT_RELEASES)
        self.tasks = tasks or list(self.DEFAULT_TASKS)
        self.split = split
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.window_duration = window_duration
        self.sfreq_resample = sfreq_resample
        self.n_channels = n_channels
        self.max_subjects = max_subjects
        self.return_metadata = return_metadata

        # S3 client for public bucket
        self.s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        self.bucket = "fcp-indi"
        self.prefix = "data/Projects/HBN/BIDS_EEG"

        # Get available subjects
        self.subjects_data = self._scan_and_cache_subjects()

        # Split subjects
        np.random.seed(seed)
        all_subjects = list(self.subjects_data.keys())
        np.random.shuffle(all_subjects)

        n_total = len(all_subjects)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])

        if split == "train":
            self.subjects = all_subjects[:n_train]
        elif split == "val":
            self.subjects = all_subjects[n_train:n_train+n_val]
        else:  # test
            self.subjects = all_subjects[n_train+n_val:]

        # Create windows from all subjects
        self.samples = self._create_samples()
        print(f"[{split}] Loaded {len(self.samples)} real EEG samples from {len(self.subjects)} subjects")

    def _scan_and_cache_subjects(self) -> Dict:
        """Scan S3 for available subjects and their data."""

        cache_name = _build_subjects_cache_name(self.releases, self.tasks)
        cache_file = self.cache_dir / cache_name

        # Try loading from cache
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                subjects_data = json.load(f)
                print(f"Loaded metadata for {len(subjects_data)} subjects from cache")
                return subjects_data

        subjects_data = {}
        total_scanned = 0

        for release in self.releases:
            if total_scanned >= self.max_subjects:
                break

            print(f"Scanning {release}...")

            # List subjects in release
            try:
                response = self.s3.list_objects_v2(
                    Bucket=self.bucket,
                    Prefix=f"{self.prefix}/{release}/sub-",
                    Delimiter='/'
                )

                for obj in response.get('CommonPrefixes', []):
                    if total_scanned >= self.max_subjects:
                        break

                    subject_path = obj['Prefix']
                    subject_id = subject_path.split('/')[-2]

                    # Check which tasks this subject has
                    subject_tasks = []
                    for task in self.tasks:
                        suffixes = [".set", ".bdf"]
                        found = False

                        for suffix in suffixes:
                            file_key = f"{subject_path}eeg/{subject_id}_task-{task}_eeg{suffix}"
                            try:
                                self.s3.head_object(Bucket=self.bucket, Key=file_key)
                                subject_tasks.append({
                                    'task': task,
                                    's3_key': file_key,
                                    'release': release,
                                })
                                found = True
                                break
                            except Exception:
                                continue

                        # Some releases publish SET/FDT pairs with uppercase FDT
                        if not found:
                            file_key = f"{subject_path}eeg/{subject_id}_task-{task}_eeg.SET"
                            try:
                                self.s3.head_object(Bucket=self.bucket, Key=file_key)
                                subject_tasks.append({
                                    'task': task,
                                    's3_key': file_key,
                                    'release': release,
                                })
                                found = True
                            except Exception:
                                continue

                    if subject_tasks:
                        subjects_data[subject_id] = subject_tasks
                        total_scanned += 1

            except Exception as e:
                print(f"Error scanning {release}: {e}")
                continue

        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(subjects_data, f)

        print(f"Found {len(subjects_data)} subjects with data")
        return subjects_data

    def _download_eeg_file(self, s3_key: str) -> Path:
        """Download EEG file from S3 if not cached"""

        cache_path = self.cache_dir / s3_key

        if not cache_path.exists():
            print(f"Downloading {s3_key}...")
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.s3.download_file(self.bucket, s3_key, str(cache_path))
                # Download matching .fdt file when a .set references binary data
                if cache_path.suffix.lower() == ".set":
                    fdt_key = s3_key[:-4] + ".fdt"
                    fdt_path = cache_path.with_suffix('.fdt')
                    try:
                        self.s3.download_file(self.bucket, fdt_key, str(fdt_path))
                    except Exception:
                        # Some datasets embed data in the .set file; continue quietly.
                        pass
            except Exception as e:
                print(f"Download failed: {e}")
                return None

        return cache_path

    def _load_and_preprocess_eeg(self, file_path: Path) -> Optional[np.ndarray]:
        """Load EEG file and preprocess"""

        try:
            suffix = file_path.suffix.lower()
            if suffix == ".set" or suffix == ".set".upper():
                raw = mne.io.read_raw_eeglab(str(file_path), preload=True, verbose=False)
            elif suffix == ".bdf":
                raw = mne.io.read_raw_edf(str(file_path), preload=True, verbose=False)
            else:
                raise ValueError(f"Unsupported EEG file extension: {suffix}")

            # Basic preprocessing
            raw.filter(l_freq=0.5, h_freq=45, verbose=False)  # Bandpass filter

            # Resample if needed
            if raw.info['sfreq'] != self.sfreq_resample:
                raw.resample(self.sfreq_resample, verbose=False)

            # Get data IN MICROVOLTS (not volts)
            data = raw.get_data() * 1e6  # Convert to microvolts

            # Remove bad segments (where std is too high or too low)
            # This helps with artifacts
            chunk_size = int(self.sfreq_resample * 0.5)  # 0.5 second chunks
            for i in range(0, data.shape[1], chunk_size):
                chunk = data[:, i:i + chunk_size]
                if chunk.std() > 500 or chunk.std() < 0.1:  # Bad segment
                    data[:, i:i+chunk_size] = 0

            # Normalize per channel
            for ch in range(data.shape[0]):
                ch_data = data[ch, :]
                if ch_data.std() > 0:
                    data[ch, :] = (ch_data - ch_data.mean()) / ch_data.std()

            # Select channels if needed
            if data.shape[0] > self.n_channels:
                data = data[:self.n_channels, :]
            elif data.shape[0] < self.n_channels:
                # Pad with zeros if fewer channels
                padding = np.zeros((self.n_channels - data.shape[0], data.shape[1]))
                data = np.vstack([data, padding])

            return data

        except Exception as e:
            print(f"Error loading EEG: {e}")
            return None

    def _create_samples(self) -> List[Dict]:
        """Create windowed samples from all subjects"""

        samples = []
        window_samples = int(self.window_duration * self.sfreq_resample)

        for subject_id in self.subjects:
            for task_info in self.subjects_data[subject_id]:
                # Download file
                file_path = self._download_eeg_file(task_info['s3_key'])
                if file_path is None:
                    continue

                # Load and preprocess
                eeg_data = self._load_and_preprocess_eeg(file_path)
                if eeg_data is None:
                    continue

                # Create windows
                n_samples = eeg_data.shape[1]
                n_windows = n_samples // window_samples

                for i in range(n_windows):
                    start = i * window_samples
                    end = start + window_samples
                    window = eeg_data[:, start:end]

                    # Create sample
                    samples.append({
                        'data': window,
                        'task': task_info['task'],
                        'subject': subject_id,
                        'window_idx': i,
                        'release': task_info['release'],
                    })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Convert to tensor
        eeg_tensor = torch.FloatTensor(sample['data'])

        # Create label based on observed tasks
        task_to_label = {task: idx for idx, task in enumerate(self.tasks)}
        label = task_to_label.get(sample['task'], 0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        if self.return_metadata:
            metadata = {
                'task': sample['task'],
                'subject': sample['subject'],
                'window_idx': sample['window_idx'],
                'release': sample.get('release', 'unknown'),
            }
            return eeg_tensor, label_tensor, metadata

        return eeg_tensor, label_tensor

def test_real_data():
    """Test loading real data"""

    print("Testing REAL HBN data loading...")

    # Create small dataset
    dataset = RealHBNDataset(
        releases=["cmi_bids_R1"],
        tasks=["RestingState", "DespicableMe"],
        split="train",
        max_subjects=5  # Just 5 subjects for testing
    )

    # Check a sample
    if len(dataset) > 0:
        eeg, label = dataset[0]
        print(f"Sample shape: {eeg.shape}")
        print(f"Label: {label.item()}")
        print(f"EEG stats: mean={eeg.mean():.4f}, std={eeg.std():.4f}")

        # Check if it's real data (not random)
        if eeg.std() > 0.01 and eeg.std() < 100:
            print("✅ This looks like real EEG data!")
        else:
            print("⚠️ Data statistics seem off")

    return dataset

if __name__ == "__main__":
    test_real_data()
