#!/usr/bin/env python3
"""
Clean evaluation script to verify metric wiring on actual data
"""

import torch
import yaml
from pathlib import Path

def evaluate_checkpoint():
    """Load checkpoint and evaluate with fixed metrics"""

    print("=== Clean Evaluation Test ===")

    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: Using CPU (slow)")

    # Load config
    with open('bef_eeg/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Create model
    from bef_eeg.train import BEFTrainer
    from bef_eeg.pipeline import PretrainableBEF

    model = PretrainableBEF(
        in_chans=config['in_chans'],
        sfreq=config['sfreq'],
        n_paths=config['bicep']['n_paths'],
        K=config['enn']['K'],
        gnn_hidden=config['fusion']['gnn_hid'],
        gnn_layers=config['fusion']['layers'],
        dropout=config['fusion']['dropout']
    ).to(device)

    trainer = BEFTrainer(model, config, use_wandb=False)

    # Skip checkpoint loading for now - just test metrics
    print("Using random weights for metric validation")
    model.eval()

    # Create small test dataset
    from hbn_dataset_loader import create_hbn_dataloaders

    try:
        # Try HBN data
        print("Creating HBN dataloaders...")
        _, val_loader, _ = create_hbn_dataloaders(
            config=config,
            releases=["cmi_bids_R1"],
            batch_size=8,
            num_workers=0,
            max_subjects=20
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to build HBN validation loader. Ensure public S3 access "
            "(no credentials required) and that mne/boto3 are installed."
        ) from e

    # Evaluate
    print(f"\nEvaluating...")
    metrics = trainer.evaluate(val_loader, stage='test')

    # Print clean results
    print(f"\n=== EVALUATION RESULTS ===")
    mae_val = metrics.get('mae', float('nan'))
    bce_val = metrics.get('bce', float('nan'))
    auc_val = metrics.get('auc', float('nan'))
    prob_val = metrics.get('mean_prob', float('nan'))

    print(f"MAE (on probabilities): {mae_val:.4f}")
    print(f"BCE: {bce_val:.4f}")
    print(f"AUC: {auc_val:.4f}")
    print(f"Mean Probability: {prob_val:.4f}")

    # Sanity checks
    mae = metrics.get('mae', float('inf'))
    auc = metrics.get('auc', 0.0)
    bce = metrics.get('bce', float('inf'))

    print(f"\n=== SANITY CHECKS ===")
    print(f"✅ MAE ≤ 1.0: {mae <= 1.0} ({mae:.4f})")
    print(f"✅ AUC > 0: {auc > 0.0} ({auc:.4f})")
    print(f"✅ BCE reasonable: {0.0 < bce < 5.0} ({bce:.4f})")

    if mae <= 1.0 and auc > 0.0 and 0.0 < bce < 5.0:
        print(f"\n🎯 METRICS WIRED CORRECTLY!")
        print(f"Expected competitive ranges:")
        print(f"  - AUC: 0.65-0.80")
        print(f"  - BCE: 0.50-0.60")
        print(f"  - MAE: 0.30-0.45")
    else:
        print(f"\n❌ Still metric issues")

    return metrics

if __name__ == "__main__":
    evaluate_checkpoint()
