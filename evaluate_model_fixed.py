'''
Evaluate the trained BEF model with proper binary metrics
'''

import yaml
import numpy as np
import torch

from binary_evaluation import compute_calibration, eval_binary
from bef_eeg.pipeline import PretrainableBEF
from hbn_dataset_loader import create_hbn_dataloaders


def evaluate_checkpoint(
    checkpoint_path: str = "bef_eeg/weights_challenge_1.pt",
    releases = ("cmi_bids_R1", "cmi_bids_R2"),
    max_subjects: int = 40,
):
    """Evaluate a trained checkpoint on real HBN EEG data."""

    print(f"Loading checkpoint from {checkpoint_path}...")

    # Load config
    with open("optimized_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PretrainableBEF(
        in_chans=config["in_chans"],
        sfreq=config["sfreq"],
        n_paths=config["bicep"]["n_paths"],
        K=config["enn"]["K"],
        gnn_hidden=config["fusion"]["gnn_hid"],
        gnn_layers=config["fusion"]["layers"],
        dropout=config["fusion"]["dropout"],
        output_dim=1,
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully!")

    print("\nSampling validation windows from real HBN releases...")
    _, val_loader, _ = create_hbn_dataloaders(
        config=config,
        releases=releases,
        batch_size=32,
        num_workers=0,
        tasks=["RestingState", "DespicableMe"],
        max_subjects=max_subjects,
    )

    probs_collector = []
    target_collector = []

    with torch.no_grad():
        for eeg, label in val_loader:
            eeg = eeg.to(device)
            label = label.to(device)

            outputs = model(eeg)
            logits = outputs["prediction"].squeeze(-1)
            probs = torch.sigmoid(logits)

            probs_collector.append(probs.cpu().numpy())
            # Treat RestingState (0) vs DespicableMe (1) as binary outcome
            target_collector.append((label == 1).long().cpu().numpy())

    if not probs_collector:
        raise RuntimeError("Validation loader produced no samples; check S3 availability.")

    all_probs = np.concatenate(probs_collector)
    targets = np.concatenate(target_collector)

    # Compute metrics
    print('\nComputing binary classification metrics...')
    metrics = eval_binary(all_probs, targets)
    cal_metrics = compute_calibration(all_probs, targets)
    
    # Print results
    print('\n' + '='*60)
    print('BINARY CLASSIFICATION EVALUATION RESULTS')
    print('='*60)
    
    # Key metrics comparison with baseline
    baseline_brier = metrics['pos_rate'] * (1 - metrics['pos_rate'])
    
    print(f'\n1. Class Balance:')
    print(f'   Positive rate: {metrics["pos_rate"]:.3f}')
    print(f'   Baseline Brier (random): {baseline_brier:.4f}')
    
    print(f'\n2. Probability Calibration:')
    print(f'   Brier Score: {metrics["brier_score"]:.4f} (lower is better)')
    print(f'   Brier Skill: {metrics["brier_skill"]:.3f} ({metrics["brier_skill"]*100:.1f}% better than random)')
    print(f'   ECE: {cal_metrics["ece"]:.4f} (expected calibration error)')
    print(f'   Mean Predicted Prob: {metrics["mean_prob"]:.3f}')
    print(f'   Std Predicted Prob: {metrics["std_prob"]:.3f}')
    
    print(f'\n3. Classification Performance:')
    print(f'   ROC-AUC: {metrics["roc_auc"]:.4f} (random=0.500)')
    print(f'   PR-AUC: {metrics["pr_auc"]:.4f} (baseline={metrics["pos_rate"]:.3f})')
    
    print(f'\n4. At Default Threshold (0.5):')
    print(f'   Accuracy: {metrics["acc@0.5"]:.4f}')
    print(f'   F1 Score: {metrics["f1@0.5"]:.4f}')
    
    print(f'\n5. At Optimal Threshold ({metrics["best_thr"]:.3f}):')
    print(f'   Accuracy: {metrics["acc@best"]:.4f} (Δ={100*(metrics["acc@best"]-metrics["acc@0.5"]):.1f}%)')
    print(f'   F1 Score: {metrics["f1@best"]:.4f} (Δ={100*(metrics["f1@best"]-metrics["f1@0.5"]):.1f}%)')
    print(f'   Sensitivity (TPR): {metrics["tpr@best"]:.3f}')
    print(f'   Specificity (1-FPR): {1-metrics["fpr@best"]:.3f}')
    
    print(f'\n6. Confusion Matrix @ Optimal Threshold:')
    print(f'                 Predicted')
    print(f'                 Neg    Pos')
    print(f'   Actual Neg   {metrics["tn"]:4d}   {metrics["fp"]:4d}')
    print(f'   Actual Pos   {metrics["fn"]:4d}   {metrics["tp"]:4d}')
    
    return metrics


if __name__ == '__main__':
    evaluate_checkpoint()
