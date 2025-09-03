"""
Extract real reaction time targets from BIDS events.tsv files
Replaces random targets with actual behavioral data
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import mne
    from mne_bids import get_bids_path_from_fname
    HAS_BIDS = True
except ImportError:
    HAS_BIDS = False
    print("Warning: MNE-BIDS not available")


def parse_events_file(events_tsv_path: str) -> pd.DataFrame:
    """
    Parse BIDS events.tsv file to extract trial information
    
    Args:
        events_tsv_path: Path to events.tsv file
        
    Returns:
        DataFrame with trial-level information
    """
    
    if not os.path.exists(events_tsv_path):
        raise FileNotFoundError(f"Events file not found: {events_tsv_path}")
        
    # Read events file
    events_df = pd.read_csv(events_tsv_path, sep='\t')
    
    # Ensure onset is numeric
    events_df['onset'] = pd.to_numeric(events_df['onset'], errors='coerce')
    events_df = events_df.sort_values('onset', kind='mergesort').reset_index(drop=True)
    
    return events_df


def extract_contrast_trials(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract contrast change detection trials with reaction times
    
    Args:
        events_df: Raw events DataFrame
        
    Returns:
        DataFrame with one row per contrast trial
    """
    
    # Find trial boundaries
    trials = events_df[events_df['value'].eq('contrastTrial_start')].copy()
    stimuli = events_df[events_df['value'].isin(['left_target', 'right_target'])].copy()
    responses = events_df[events_df['value'].isin(['left_buttonPress', 'right_buttonPress'])].copy()
    
    trials = trials.reset_index(drop=True)
    trials['next_onset'] = trials['onset'].shift(-1)
    trials = trials.dropna(subset=['next_onset']).reset_index(drop=True)
    
    trial_data = []
    
    for _, trial in trials.iterrows():
        start_time = float(trial['onset'])
        end_time = float(trial['next_onset'])
        
        # Find stimulus in this trial
        trial_stimuli = stimuli[
            (stimuli['onset'] >= start_time) & 
            (stimuli['onset'] < end_time)
        ]
        
        if trial_stimuli.empty:
            stimulus_onset = np.nan
            target_side = None
        else:
            stimulus_onset = float(trial_stimuli.iloc[0]['onset'])
            target_side = trial_stimuli.iloc[0]['value']
            
        # Find response in this trial
        if not np.isnan(stimulus_onset):
            trial_responses = responses[
                (responses['onset'] >= stimulus_onset) & 
                (responses['onset'] < end_time)
            ]
        else:
            trial_responses = responses[
                (responses['onset'] >= start_time) & 
                (responses['onset'] < end_time)
            ]
            
        if trial_responses.empty:
            response_onset = np.nan
            response_side = None
            feedback = None
        else:
            response_onset = float(trial_responses.iloc[0]['onset'])
            response_side = trial_responses.iloc[0]['value']
            feedback = trial_responses.iloc[0].get('feedback', None)
            
        # Calculate reaction times
        rt_from_stimulus = np.nan
        rt_from_trial = np.nan
        
        if not np.isnan(response_onset):
            rt_from_trial = response_onset - start_time
            if not np.isnan(stimulus_onset):
                rt_from_stimulus = response_onset - stimulus_onset
                
        # Determine correctness
        correct = None
        if feedback is not None:
            if feedback == 'smiley_face':
                correct = True
            elif feedback == 'sad_face':
                correct = False
        elif target_side is not None and response_side is not None:
            # Manual correctness check based on sides
            correct = (
                (target_side == 'left_target' and response_side == 'left_buttonPress') or
                (target_side == 'right_target' and response_side == 'right_buttonPress')
            )
            
        trial_data.append({
            'trial_start': start_time,
            'trial_end': end_time,
            'stimulus_onset': stimulus_onset,
            'response_onset': response_onset,
            'rt_from_stimulus': rt_from_stimulus,
            'rt_from_trial': rt_from_trial,
            'target_side': target_side,
            'response_side': response_side,
            'correct': correct,
            'feedback': feedback
        })
        
    return pd.DataFrame(trial_data)


def extract_surround_suppression_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract surround suppression events (passive viewing)
    
    Args:
        events_df: Raw events DataFrame
        
    Returns:
        DataFrame with stimulus events
    """
    
    # For SuS, we mainly care about visual stimuli
    stimulus_events = events_df[
        events_df['value'].str.contains('stimulus|visual', case=False, na=False)
    ].copy()
    
    if stimulus_events.empty:
        # If no explicit stimulus events, use all non-response events
        stimulus_events = events_df[
            ~events_df['value'].str.contains('response|button', case=False, na=False)
        ].copy()
        
    # Add some derived features for passive task
    stimulus_events['event_duration'] = stimulus_events.get('duration', 0.5)  # Default 500ms
    stimulus_events['stimulus_type'] = stimulus_events['value']
    
    return stimulus_events


def create_target_mapping(
    trial_df: pd.DataFrame,
    target_type: str = 'rt_from_stimulus',
    window_size_s: float = 2.0,
    window_stride_s: float = 1.0,
    sfreq: float = 100.0
) -> Dict[float, float]:
    """
    Create mapping from time windows to target values
    
    Args:
        trial_df: Trial DataFrame with target values
        target_type: Which target to use ('rt_from_stimulus', 'correct', etc.)
        window_size_s: Window size in seconds
        window_stride_s: Window stride in seconds  
        sfreq: Sampling frequency
        
    Returns:
        Dictionary mapping window start times to target values
    """
    
    target_mapping = {}
    
    for _, trial in trial_df.iterrows():
        if pd.isna(trial.get(target_type)):
            continue
            
        trial_start = trial['trial_start']
        trial_end = trial.get('trial_end', trial_start + window_size_s)
        target_value = trial[target_type]
        
        # Create windows for this trial
        window_start = trial_start
        while window_start + window_size_s <= trial_end:
            target_mapping[window_start] = float(target_value)
            window_start += window_stride_s
            
    return target_mapping


def process_subject_data(
    subject_dir: str,
    task: str = 'contrastChangeDetection',
    target_type: str = 'rt_from_stimulus'
) -> Tuple[Dict[float, float], pd.DataFrame]:
    """
    Process a single subject's data to extract targets
    
    Args:
        subject_dir: Path to subject directory (e.g., sub-NDARXXX)
        task: Task name ('contrastChangeDetection' or 'surroundSupp')
        target_type: Target variable to extract
        
    Returns:
        Tuple of (target_mapping, trial_dataframe)
    """
    
    subject_id = os.path.basename(subject_dir)
    
    # Find EEG files for this task
    eeg_pattern = os.path.join(subject_dir, 'eeg', f'{subject_id}_task-{task}_*.set')
    eeg_files = glob.glob(eeg_pattern)
    
    if not eeg_files:
        raise FileNotFoundError(f"No EEG files found for {subject_id} task {task}")
        
    all_targets = {}
    all_trials = []
    
    for eeg_file in eeg_files:
        # Find corresponding events file using BIDS naming
        events_file = eeg_file.replace('_eeg.set', '_events.tsv')
        
        if events_file is None or not os.path.exists(events_file):
            print(f"Warning: No events file found for {eeg_file}")
            continue
            
        # Parse events
        events_df = parse_events_file(events_file)
        
        # Extract trials based on task
        if task == 'contrastChangeDetection':
            trial_df = extract_contrast_trials(events_df)
        elif task == 'surroundSupp':
            trial_df = extract_surround_suppression_events(events_df)
        else:
            raise ValueError(f"Unknown task: {task}")
            
        # Create target mapping
        target_mapping = create_target_mapping(trial_df, target_type)
        all_targets.update(target_mapping)
        all_trials.append(trial_df)
        
    # Combine all trials for this subject
    if all_trials:
        combined_trials = pd.concat(all_trials, ignore_index=True)
    else:
        combined_trials = pd.DataFrame()
        
    return all_targets, combined_trials


def extract_targets_for_dataset(
    data_dir: str,
    task: str = 'contrastChangeDetection', 
    target_type: str = 'rt_from_stimulus',
    max_subjects: Optional[int] = None
) -> Dict[str, Dict[float, float]]:
    """
    Extract targets for entire dataset
    
    Args:
        data_dir: Root data directory
        task: Task name
        target_type: Target variable
        max_subjects: Limit number of subjects
        
    Returns:
        Dictionary mapping subject IDs to their target mappings
    """
    
    # Find all subject directories
    subject_dirs = glob.glob(os.path.join(data_dir, 'sub-*'))
    subject_dirs = [d for d in subject_dirs if os.path.isdir(d)]
    
    if max_subjects:
        subject_dirs = subject_dirs[:max_subjects]
        
    print(f"Processing {len(subject_dirs)} subjects for task {task}")
    
    dataset_targets = {}
    
    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        
        try:
            target_mapping, trial_df = process_subject_data(
                subject_dir, task, target_type
            )
            
            dataset_targets[subject_id] = target_mapping
            print(f"✓ {subject_id}: {len(target_mapping)} target windows, "
                  f"{len(trial_df)} trials")
                  
        except Exception as e:
            print(f"✗ {subject_id}: Error - {e}")
            continue
            
    return dataset_targets


def save_targets(
    targets_dict: Dict[str, Dict[float, float]], 
    output_path: str,
    task: str,
    target_type: str
):
    """Save extracted targets to file"""
    
    # Convert to DataFrame for easy saving
    rows = []
    for subject_id, target_mapping in targets_dict.items():
        for window_time, target_value in target_mapping.items():
            rows.append({
                'subject_id': subject_id,
                'window_start_time': window_time,
                'target_value': target_value,
                'task': task,
                'target_type': target_type
            })
            
    df = pd.DataFrame(rows)
    
    # Save as CSV and pickle
    df.to_csv(output_path.replace('.pkl', '.csv'), index=False)
    df.to_pickle(output_path)
    
    print(f"Saved {len(rows)} target mappings to {output_path}")


def test_target_extraction():
    """Test target extraction on R5 mini dataset"""
    
    data_dir = "/home/ryan/research/eeg_challenge/R5_mini_L100-20250903T052429Z-1-001/R5_mini_L100"
    
    if not os.path.exists(data_dir):
        print(f"Test data directory not found: {data_dir}")
        return
        
    print("Testing target extraction...")
    
    # Extract CCD targets
    ccd_targets = extract_targets_for_dataset(
        data_dir=data_dir,
        task='contrastChangeDetection',
        target_type='rt_from_stimulus',
        max_subjects=3  # Test with just 3 subjects
    )
    
    # Extract SuS targets (for pretraining)
    sus_targets = extract_targets_for_dataset(
        data_dir=data_dir,
        task='surroundSupp', 
        target_type='stimulus_type',  # Different target for passive task
        max_subjects=3
    )
    
    print(f"CCD targets extracted: {len(ccd_targets)} subjects")
    print(f"SuS targets extracted: {len(sus_targets)} subjects")
    
    # Save for inspection
    if ccd_targets:
        save_targets(ccd_targets, '/tmp/ccd_targets.pkl', 'ccd', 'rt_from_stimulus')
    if sus_targets:  
        save_targets(sus_targets, '/tmp/sus_targets.pkl', 'sus', 'stimulus_type')


if __name__ == "__main__":
    test_target_extraction()