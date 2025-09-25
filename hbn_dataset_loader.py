#!/usr/bin/env python3
"""Real HBN dataset helpers.

This module exposes a thin wrapper around :class:`RealHBNDataset` so that any
script requesting train/validation/test loaders always touches the official
Child Mind Institute releases hosted on ``s3://fcp-indi/data/Projects/HBN``.
All synthetic fallbacks have been removed.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from torch.utils.data import DataLoader

from real_hbn_loader import RealHBNDataset


def create_hbn_dataloaders(
    config: Dict,
    releases: Optional[Iterable[str]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    tasks: Optional[Iterable[str]] = None,
    max_subjects: Optional[int] = None,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Instantiate train/val/test loaders that stream the real HBN EEG data."""

    from bef_eeg.utils_io import (
        _resolve_cache_dir,
        _resolve_max_subjects,
        _resolve_releases,
    )

    resolved_releases = _resolve_releases(config, releases)
    resolved_cache = _resolve_cache_dir(config, cache_dir)
    resolved_max_subjects = _resolve_max_subjects(config, max_subjects, resolved_releases)
    resolved_tasks = list(tasks) if tasks else list(RealHBNDataset.DEFAULT_TASKS)

    dataset_kwargs = dict(
        releases=resolved_releases,
        tasks=resolved_tasks,
        split_ratios=split_ratios,
        max_subjects=resolved_max_subjects,
        cache_dir=str(resolved_cache),
        sfreq_resample=int(config.get("sfreq", 100)),
        n_channels=int(config.get("in_chans", 129)),
        seed=seed,
    )

    def _make(split: str, shuffle: bool) -> DataLoader:
        dataset = RealHBNDataset(split=split, **dataset_kwargs)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    return _make("train", True), _make("val", False), _make("test", False)


__all__ = ["create_hbn_dataloaders"]
