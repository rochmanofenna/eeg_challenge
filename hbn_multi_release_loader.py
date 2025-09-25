#!/usr/bin/env python3
"""Compatibility wrapper for multi-release HBN loading.

Historically this module returned synthetic tensors to mimic the public EEG
challenge. It now proxies ``create_hbn_dataloaders`` so that every consumer
receives real recordings streamed from the Child Mind Institute S3 releases.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

from torch.utils.data import DataLoader

from hbn_dataset_loader import create_hbn_dataloaders


def create_hbn_multi_release_loaders(
    config: Dict,
    batch_size: int = 32,
    num_workers: int = 4,
    releases: Optional[Iterable[str]] = None,
    tasks: Optional[Iterable[str]] = None,
    max_subjects: Optional[int] = None,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Alias for :func:`hbn_dataset_loader.create_hbn_dataloaders`."""

    return create_hbn_dataloaders(
        config=config,
        releases=releases,
        batch_size=batch_size,
        num_workers=num_workers,
        tasks=tasks,
        max_subjects=max_subjects,
        split_ratios=split_ratios,
    )


__all__ = ["create_hbn_multi_release_loaders"]
