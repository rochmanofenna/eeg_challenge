"""
Integration with EEG Challenge startkit for real data loading
Combines the startkit's preprocessing with our ENN pipeline
"""

import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List

# Add startkit to path
sys.path.append('/home/ryan/research/eeg_challenge/startkit')

# Import startkit components
try:
    import pandas as pd
    from eegdash.dataset import EEGChallengeDataset
    import mne
    from mne_bids import get_bids_path_from_fname
    from braindecode.datasets import BaseConcatDataset, BaseDataset
    from braindecode.preprocessing import preprocess, Preprocessor, create_windows_from_events
    from sklearn.model_selection import train_test_split
    from sklearn.utils import check_random_state
    from joblib import Parallel, delayed
    HAS_STARTKIT = True
    print("Successfully imported startkit components")
except ImportError as e:
    HAS_STARTKIT = False
    print(f"Warning: Could not import startkit components: {e}")
    print("Using mock data instead")


def build_trial_table(events_df: pd.DataFrame) -> pd.DataFrame:
    """One row per contrast trial with stimulus/response metrics."""
    events_df = events_df.copy()
    events_df["onset"] = pd.to_numeric(events_df["onset"], errors="raise")
    events_df = events_df.sort_values("onset", kind="mergesort").reset_index(drop=True)

    trials = events_df[events_df["value"].eq("contrastTrial_start")].copy()
    stimuli = events_df[events_df["value"].isin(["left_target", "right_target"])].copy()
    responses = events_df[events_df["value"].isin(["left_buttonPress", "right_buttonPress"])].copy()

    trials = trials.reset_index(drop=True)
    trials["next_onset"] = trials["onset"].shift(-1)
    trials = trials.dropna(subset=["next_onset"]).reset_index(drop=True)

    rows = []
    for _, tr in trials.iterrows():
        start = float(tr["onset"])
        end = float(tr["next_onset"])

        stim_block = stimuli[(stimuli["onset"] >= start) & (stimuli["onset"] < end)]
        stim_onset = np.nan if stim_block.empty else float(stim_block.iloc[0]["onset"])

        if not np.isnan(stim_onset):
            resp_block = responses[(responses["onset"] >= stim_onset) & (responses["onset"] < end)]
        else:
            resp_block = responses[(responses["onset"] >= start) & (responses["onset"] < end)]

        if resp_block.empty:
            resp_onset = np.nan
            resp_type = None
            feedback = None
        else:
            resp_onset = float(resp_block.iloc[0]["onset"])
            resp_type = resp_block.iloc[0]["value"]
            feedback = resp_block.iloc[0]["feedback"]

        rt_from_stim = (resp_onset - stim_onset) if (not np.isnan(stim_onset) and not np.isnan(resp_onset)) else np.nan
        rt_from_trial = (resp_onset - start) if not np.isnan(resp_onset) else np.nan

        correct = None
        if isinstance(feedback, str):
            if feedback == "smiley_face": correct = True
            elif feedback == "sad_face": correct = False

        rows.append({
            "trial_start_onset": start,
            "trial_stop_onset": end,
            "stimulus_onset": stim_onset,
            "response_onset": resp_onset,
            "rt_from_stimulus": rt_from_stim,
            "rt_from_trialstart": rt_from_trial,
            "response_type": resp_type,
            "correct": correct,
        })

    return pd.DataFrame(rows)


def _to_float_or_none(x):
    return None if pd.isna(x) else float(x)


def _to_int_or_none(x):
    if pd.isna(x):
        return None
    if isinstance(x, (bool, np.bool_)):
        return int(bool(x))
    if isinstance(x, (int, np.integer)):
        return int(x)
    try:
        return int(x)
    except Exception:
        return None


def _to_str_or_none(x):
    return None if (x is None or (isinstance(x, float) and np.isnan(x))) else str(x)


def annotate_trials_with_target(raw, target_field="rt_from_stimulus", epoch_length=2.0,
                                require_stimulus=True, require_response=True):
    """Create 'contrast_trial_start' annotations with float target in extras."""
    fnames = raw.filenames
    assert len(fnames) == 1, "Expected a single filename"
    bids_path = get_bids_path_from_fname(fnames[0])
    events_file = bids_path.update(suffix="events", extension=".tsv").fpath

    events_df = (pd.read_csv(events_file, sep="\t")
                   .assign(onset=lambda d: pd.to_numeric(d["onset"], errors="raise"))
                   .sort_values("onset", kind="mergesort").reset_index(drop=True))

    trials = build_trial_table(events_df)

    if require_stimulus:
        trials = trials[trials["stimulus_onset"].notna()].copy()
    if require_response:
        trials = trials[trials["response_onset"].notna()].copy()

    if target_field not in trials.columns:
        raise KeyError(f"{target_field} not in computed trial table.")
    targets = trials[target_field].astype(float)

    onsets = trials["trial_start_onset"].to_numpy(float)
    durations = np.full(len(trials), float(epoch_length), dtype=float)
    descs = ["contrast_trial_start"] * len(trials)

    extras = []
    for i, v in enumerate(targets):
        row = trials.iloc[i]

        extras.append({
            "target": _to_float_or_none(v),
            "rt_from_stimulus": _to_float_or_none(row["rt_from_stimulus"]),
            "rt_from_trialstart": _to_float_or_none(row["rt_from_trialstart"]),
            "stimulus_onset": _to_float_or_none(row["stimulus_onset"]),
            "response_onset": _to_float_or_none(row["response_onset"]),
            "correct": _to_int_or_none(row["correct"]),
            "response_type": _to_str_or_none(row["response_type"]),
        })

    new_ann = mne.Annotations(onset=onsets, duration=durations, description=descs,
                              orig_time=raw.info["meas_date"], extras=extras)
    raw.set_annotations(new_ann, verbose=False)
    return raw


def add_aux_anchors(raw, stim_desc="stimulus_anchor", resp_desc="response_anchor"):
    ann = raw.annotations
    mask = (ann.description == "contrast_trial_start")
    if not np.any(mask):
        return raw

    stim_onsets, resp_onsets = [], []
    stim_extras, resp_extras = [], []

    for idx in np.where(mask)[0]:
        ex = ann.extras[idx] if ann.extras is not None else {}
        t0 = float(ann.onset[idx])

        stim_t = ex["stimulus_onset"]
        resp_t = ex["response_onset"]

        if stim_t is None or (isinstance(stim_t, float) and np.isnan(stim_t)):
            rtt = ex["rt_from_trialstart"]
            rts = ex["rt_from_stimulus"]
            if rtt is not None and rts is not None:
                stim_t = t0 + float(rtt) - float(rts)

        if resp_t is None or (isinstance(resp_t, float) and np.isnan(resp_t)):
            rtt = ex["rt_from_trialstart"]
            if rtt is not None:
                resp_t = t0 + float(rtt)

        if (stim_t is not None) and not (isinstance(stim_t, float) and np.isnan(stim_t)):
            stim_onsets.append(float(stim_t))
            stim_extras.append(dict(ex, anchor="stimulus"))
        if (resp_t is not None) and not (isinstance(resp_t, float) and np.isnan(resp_t)):
            resp_onsets.append(float(resp_t))
            resp_extras.append(dict(ex, anchor="response"))

    new_onsets = np.array(stim_onsets + resp_onsets, dtype=float)
    if len(new_onsets):
        aux = mne.Annotations(
            onset=new_onsets,
            duration=np.zeros_like(new_onsets, dtype=float),
            description=[stim_desc]*len(stim_onsets) + [resp_desc]*len(resp_onsets),
            orig_time=raw.info["meas_date"],
            extras=stim_extras + resp_extras,
        )
        raw.set_annotations(ann + aux, verbose=False)
    return raw


def add_extras_columns(
    windows_concat_ds,
    original_concat_ds,
    desc="contrast_trial_start",
    keys=("target","rt_from_stimulus","rt_from_trialstart","stimulus_onset","response_onset","correct","response_type"),
):
    float_cols = {"target","rt_from_stimulus","rt_from_trialstart","stimulus_onset","response_onset"}

    for win_ds, base_ds in zip(windows_concat_ds.datasets, original_concat_ds.datasets):
        ann = base_ds.raw.annotations
        idx = np.where(ann.description == desc)[0]
        if idx.size == 0:
            continue

        per_trial = [
            {k: (ann.extras[i][k] if ann.extras is not None and k in ann.extras[i] else None) for k in keys}
            for i in idx
        ]

        md = win_ds.metadata.copy()
        first = (md["i_window_in_trial"].to_numpy() == 0)
        trial_ids = first.cumsum() - 1
        n_trials = trial_ids.max() + 1 if len(trial_ids) else 0
        assert n_trials == len(per_trial), f"Trial mismatch: {n_trials} vs {len(per_trial)}"

        for k in keys:
            vals = [per_trial[t][k] if t < len(per_trial) else None for t in trial_ids]
            if k == "correct":
                ser = pd.Series([None if v is None else int(bool(v)) for v in vals],
                                index=md.index, dtype="Int64")
            elif k in float_cols:
                ser = pd.Series([np.nan if v is None else float(v) for v in vals],
                                index=md.index, dtype="Float64")
            else:  # response_type
                ser = pd.Series(vals, index=md.index, dtype="string")

            # Replace the whole column to avoid dtype conflicts
            md[k] = ser

        win_ds.metadata = md.reset_index(drop=True)
        if hasattr(win_ds, "y"):
            y_np = win_ds.metadata["target"].astype(float).to_numpy()
            win_ds.y = y_np[:, None]  # (N, 1)

    return windows_concat_ds


def keep_only_recordings_with(desc, concat_ds):
    kept = []
    for ds in concat_ds.datasets:
        if np.any(ds.raw.annotations.description == desc):
            kept.append(ds)
        else:
            print(f"[warn] Recording {ds.raw.filenames[0]} does not contain event '{desc}'")
    return BaseConcatDataset(kept)


def create_challenge_data_loaders(
    data_dir: str = "data",
    release: str = "R5",
    mini: bool = True,
    batch_size: int = 32,
    num_workers: int = 0,
    valid_split: float = 0.1,
    test_split: float = 0.1,
    seed: int = 2025
) -> Dict[str, DataLoader]:
    """
    Create data loaders for both tasks using the startkit preprocessing
    
    Returns:
        Dictionary with train/valid/test loaders for both tasks
    """
    
    if not HAS_STARTKIT:
        print("Startkit not available, falling back to mock data")
        # Use our existing mock data approach from train.py
        from torch.utils.data import TensorDataset, DataLoader
        
        # Mock data dimensions
        n_samples = 1000
        n_channels = 129
        n_times = 200
        
        # Create random data
        X = torch.randn(n_samples, n_channels, n_times)
        y = torch.randn(n_samples, 1)  # Regression targets
        
        # Create mock info for each sample
        mock_info = [{'subject': i % 20, 'rt_from_stimulus': 1.5 + torch.randn(1).item() * 0.3, 
                     'correct': torch.rand(1).item() > 0.5} for i in range(n_samples)]
        
        # Create wrapper dataset that includes info
        class MockEEGDataset(TensorDataset):
            def __init__(self, X, y, info_list):
                super().__init__(X, y)
                self.info_list = info_list
                
            def __getitem__(self, index):
                X, y = super().__getitem__(index)
                info = self.info_list[index]
                return X, y, info
        
        # Create datasets
        dataset = MockEEGDataset(X, y, mock_info)
        
        # Split into train/valid/test
        train_size = int(0.7 * n_samples)
        valid_size = int(0.15 * n_samples)
        
        # Split indices
        indices = list(range(n_samples))
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:train_size + valid_size]
        test_indices = indices[train_size + valid_size:]
        
        # Create proper subset datasets that maintain the info structure
        class MockEEGSubset:
            def __init__(self, parent_dataset, indices):
                self.parent_dataset = parent_dataset
                self.indices = indices
                
            def __len__(self):
                return len(self.indices)
                
            def __getitem__(self, idx):
                return self.parent_dataset[self.indices[idx]]
        
        train_dataset = MockEEGSubset(dataset, train_indices)
        valid_dataset = MockEEGSubset(dataset, valid_indices)
        test_dataset = MockEEGSubset(dataset, test_indices)
        
        # Create loaders for both tasks
        loaders = {
            'ccd_train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
            'ccd_valid': DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
            'ccd_test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
            'sus_train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
            'sus_valid': DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
            'sus_test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        }
        
        return loaders
    
    print("Loading challenge datasets...")
    
    # Load datasets
    dataset_sus = EEGChallengeDataset(
        task="surroundSuppression",
        release=release,
        cache_dir=Path(data_dir),
        mini=mini
    )
    
    dataset_ccd = EEGChallengeDataset(
        task="contrastChangeDetection",
        release=release,
        cache_dir=Path(data_dir),
        mini=mini
    )
    
    print(f"Loaded {len(dataset_sus.datasets)} SuS recordings")
    print(f"Loaded {len(dataset_ccd.datasets)} CCD recordings")
    
    # Process CCD data (active task for Challenge 1)
    print("Processing CCD data...")
    
    EPOCH_LEN_S = 2.0
    SFREQ = 100
    
    transformation_offline = [
        Preprocessor(
            annotate_trials_with_target,
            target_field="rt_from_stimulus", 
            epoch_length=EPOCH_LEN_S,
            require_stimulus=True, 
            require_response=True,
            apply_on_array=False,
        ),
        Preprocessor(add_aux_anchors, apply_on_array=False),
    ]
    
    preprocess(dataset_ccd, transformation_offline, n_jobs=1)
    
    ANCHOR = "stimulus_anchor"
    SHIFT_AFTER_STIM = 0.5
    WINDOW_LEN = 2.0
    
    # Keep only recordings that actually contain stimulus anchors
    dataset_ccd = keep_only_recordings_with(ANCHOR, dataset_ccd)
    
    # Create single-interval windows (stim-locked)
    ccd_windows = create_windows_from_events(
        dataset_ccd,
        mapping={ANCHOR: 0},
        trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),
        trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * SFREQ),
        window_size_samples=int(EPOCH_LEN_S * SFREQ),
        window_stride_samples=SFREQ,
        preload=True,
    )
    
    # Inject metadata
    ccd_windows = add_extras_columns(
        ccd_windows,
        dataset_ccd,
        desc=ANCHOR,
        keys=("target", "rt_from_stimulus", "rt_from_trialstart",
              "stimulus_onset", "response_onset", "correct", "response_type")
    )
    
    # Process SuS data (passive task for pretraining)
    print("Processing SuS data...")
    
    # For SuS, we'll create simpler windows since it's a passive task
    # This is simplified - in practice you'd want proper SuS preprocessing
    sus_windows = create_windows_from_events(
        dataset_sus,
        mapping=None,  # Use all events
        window_size_samples=int(EPOCH_LEN_S * SFREQ),
        window_stride_samples=SFREQ,
        preload=True,
    )
    
    # Create train/valid/test splits
    print("Creating data splits...")
    
    # CCD splits (by subject)
    ccd_meta = ccd_windows.get_metadata()
    ccd_subjects = ccd_meta["subject"].unique()
    
    ccd_train_subj, ccd_valid_test = train_test_split(
        ccd_subjects, 
        test_size=(valid_split + test_split), 
        random_state=check_random_state(seed),
        shuffle=True
    )
    
    ccd_valid_subj, ccd_test_subj = train_test_split(
        ccd_valid_test, 
        test_size=test_split/(valid_split + test_split), 
        random_state=check_random_state(seed + 1),
        shuffle=True
    )
    
    # SuS splits (by subject)
    sus_meta = sus_windows.get_metadata()
    sus_subjects = sus_meta["subject"].unique()
    
    sus_train_subj, sus_valid_test = train_test_split(
        sus_subjects,
        test_size=(valid_split + test_split),
        random_state=check_random_state(seed),
        shuffle=True
    )
    
    sus_valid_subj, sus_test_subj = train_test_split(
        sus_valid_test,
        test_size=test_split/(valid_split + test_split),
        random_state=check_random_state(seed + 1),
        shuffle=True
    )
    
    # Create split datasets
    def split_by_subjects(windows_ds, train_subj, valid_subj, test_subj):
        subject_split = windows_ds.split("subject")
        train_set = []
        valid_set = []
        test_set = []

        for s in subject_split:
            if s in train_subj:
                train_set.append(subject_split[s])
            elif s in valid_subj:
                valid_set.append(subject_split[s])
            elif s in test_subj:
                test_set.append(subject_split[s])

        return (BaseConcatDataset(train_set), 
                BaseConcatDataset(valid_set), 
                BaseConcatDataset(test_set))
    
    ccd_train, ccd_valid, ccd_test = split_by_subjects(
        ccd_windows, ccd_train_subj, ccd_valid_subj, ccd_test_subj
    )
    
    sus_train, sus_valid, sus_test = split_by_subjects(
        sus_windows, sus_train_subj, sus_valid_subj, sus_test_subj
    )
    
    print(f"CCD splits - Train: {len(ccd_train)}, Valid: {len(ccd_valid)}, Test: {len(ccd_test)}")
    print(f"SuS splits - Train: {len(sus_train)}, Valid: {len(sus_valid)}, Test: {len(sus_test)}")
    
    # Create data loaders
    loaders = {
        'ccd_train': DataLoader(ccd_train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'ccd_valid': DataLoader(ccd_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'ccd_test': DataLoader(ccd_test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'sus_train': DataLoader(sus_train, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'sus_valid': DataLoader(sus_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        'sus_test': DataLoader(sus_test, batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    
    return loaders