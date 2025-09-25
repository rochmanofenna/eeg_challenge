"""RunPod-friendly wrappers around the real HBN dataset."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from real_hbn_loader import RealHBNDataset


def _normalize_release(release: str) -> str:
    release = release.strip()
    if release.startswith("cmi_bids_"):
        return release
    if release.lower().startswith("r"):
        suffix = release[1:].lstrip("0") or "0"
        return f"cmi_bids_R{suffix}"
    return release


class HBNDataset(Dataset):
    """Adapter that exposes RealHBNDataset samples in RunPod's dictionary form."""

    def __init__(
        self,
        release: str,
        use_mini: bool = False,  # preserved for backward compatibility
        config: Optional[Dict] = None,
        split: str = "train",
    ):
        self.release = _normalize_release(release)
        cfg = config or {}

        cache_dir = cfg.get('cache_dir', '/tmp/real_hbn_cache')
        split_ratios = cfg.get('split_ratios', (0.7, 0.15, 0.15))
        tasks: Optional[Iterable[str]] = cfg.get('tasks')
        sfreq = cfg.get('sfreq', 100)
        n_chans = cfg.get('in_chans', 129)
        max_subjects = cfg.get('max_subjects', 10**9)
        seed = cfg.get('seed', 42)

        self._dataset = RealHBNDataset(
            releases=[self.release],
            tasks=list(tasks) if tasks else None,
            split=split,
            split_ratios=split_ratios,
            max_subjects=max_subjects,
            cache_dir=cache_dir,
            sfreq_resample=sfreq,
            n_channels=n_chans,
            seed=seed,
            return_metadata=True,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict:
        eeg, label, meta = self._dataset[idx]
        return {
            'eeg': eeg,
            'label': label,
            'subject': meta['subject'],
            'task': meta['task'],
            'release': meta['release'],
            'window_idx': meta['window_idx'],
        }


def create_hbn_dataloader(
    dataset: HBNDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    **kwargs,
) -> DataLoader:
    """Create a DataLoader that returns dictionaries matching prior code."""

    def collate(batch: List[Dict]) -> Dict:
        eeg = torch.stack([item['eeg'] for item in batch])
        labels = torch.stack([item['label'] for item in batch])
        return {
            'eeg': eeg,
            'label': labels,
            'subjects': [item['subject'] for item in batch],
            'tasks': [item['task'] for item in batch],
            'release': batch[0]['release'] if batch else dataset.release,
            'window_idx': [item['window_idx'] for item in batch],
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
        **kwargs,
    )


__all__ = ["HBNDataset", "create_hbn_dataloader"]
