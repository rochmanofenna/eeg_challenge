"""Data loading utilities wired to the real HBN S3 releases.

All helpers in this module route through ``RealHBNDataset`` so that every
training or evaluation pipeline pulls true EEG recordings from the official
Child Mind Institute S3 distribution. Synthetic placeholders have been
removed to avoid unintentionally training on random tensors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from real_hbn_loader import RealHBNDataset


# Default task order when none is provided. These correspond to tasks that are
# available across the official HBN releases listed in the public challenge
# documentation. Extend this list if additional tasks are required.
DEFAULT_TASKS: Tuple[str, ...] = (
    "RestingState",
    "DespicableMe",
    "DiaryOfAWimpyKid",
)


def _resolve_releases(config: Dict, overrides: Optional[Iterable[str]]) -> List[str]:
    """Return the list of HBN releases to use for dataset construction."""

    if overrides:
        return list(overrides)

    s3_config = config.get("s3_config", {})
    releases = s3_config.get("releases")
    if not releases:
        raise ValueError(
            "No HBN releases specified. Provide them via the config's "
            "'s3_config.releases' or the `releases` argument."
        )
    return list(releases)


def _resolve_cache_dir(config: Dict, override: Optional[str]) -> Path:
    """Return the cache directory for downloaded EEG files."""

    if override:
        return Path(override)

    s3_config = config.get("s3_config", {})
    cache_dir = s3_config.get("cache_dir", "/tmp/hbn_cache")
    return Path(cache_dir)


def _resolve_max_subjects(config: Dict, override: Optional[int], releases: List[str]) -> int:
    """Choose how many subjects to scan/download across releases."""

    if override is not None:
        return override

    s3_config = config.get("s3_config", {})
    per_release = s3_config.get("max_subjects_per_release")
    if per_release is None:
        # Default to scanning every subject in the provided releases.
        return int(1e9)

    return per_release * len(releases)


def load_eeg_data(
    config: Dict,
    batch_size: int = 32,
    num_workers: int = 4,
    releases: Optional[Iterable[str]] = None,
    tasks: Optional[Iterable[str]] = None,
    max_subjects: Optional[int] = None,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders backed by the real HBN S3 data.

    Args:
        config: Full configuration dictionary containing ``s3_config`` with
            bucket, prefix, releases, cache directory, and max subject hints.
        batch_size: Batch size for all splits.
        num_workers: Number of dataloader worker processes per split.
        releases: Optional iterable of release identifiers. Defaults to the
            values in ``config['s3_config']['releases']``.
        tasks: Optional iterable of task names to include. If omitted a small
            default set is used.
        max_subjects: Optional upper bound on total subjects to scan across all
            releases. Defaults to the per-release limit declared in the config.
        split_ratios: Train/validation/test ratios applied to the discovered
            subject list.
        cache_dir: Directory used for persisted downloads. Defaults to the
            config setting or ``/tmp/hbn_cache``.
        seed: RNG seed used for deterministic subject splitting.

    Returns:
        Tuple of ``(train_loader, val_loader, test_loader)`` each pulling real
        EEG recordings from the specified HBN releases.
    """

    if not isinstance(config, dict):
        raise TypeError(
            "`config` must be a dictionary matching bef_eeg/config.yaml. "
            "Got type {}".format(type(config))
        )

    resolved_releases = _resolve_releases(config, releases)
    resolved_tasks = list(tasks) if tasks else list(DEFAULT_TASKS)
    resolved_cache = _resolve_cache_dir(config, cache_dir)
    resolved_max_subjects = _resolve_max_subjects(config, max_subjects, resolved_releases)

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

    train_dataset = RealHBNDataset(split="train", **dataset_kwargs)
    val_dataset = RealHBNDataset(split="val", **dataset_kwargs)
    test_dataset = RealHBNDataset(split="test", **dataset_kwargs)

    def _make_loader(dataset: RealHBNDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    train_loader = _make_loader(train_dataset, shuffle=True)
    val_loader = _make_loader(val_dataset, shuffle=False)
    test_loader = _make_loader(test_dataset, shuffle=False)

    return train_loader, val_loader, test_loader


__all__ = ["load_eeg_data"]
