'''
Binary Classification Evaluation for EEG Model
Proper metrics for binary classification task
'''

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, accuracy_score,
    roc_curve, precision_recall_curve, confusion_matrix,
    classification_report, brier_score_loss
)
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def eval_binary(probs: np.ndarray, targets: np.ndarray) -> Dict:
    '''
    Comprehensive binary classification evaluation
    
    Args:
        probs: Predicted probabilities [0,1]
        targets: True labels {0,1}
    
    Returns:
        Dictionary of metrics
    '''
    probs = np.asarray(probs).ravel()
    y = np.asarray(targets).astype(int).ravel()
    
    # Check class balance
    pos_rate = y.mean()
    
    # Default 0.5 threshold metrics
    preds05 = (probs >= 0.5).astype(int)
    acc05 = accuracy_score(y, preds05)
    f105 = f1_score(y, preds05)
    
    # ROC and PR curves
    try:
        auc = roc_auc_score(y, probs)
        ap = average_precision_score(y, probs)
    except:
        auc = ap = np.nan
    
    # Find optimal threshold by Youden's J (tpr - fpr)
    fpr, tpr, thr = roc_curve(y, probs)
    j = tpr - fpr
    best_idx = np.argmax(j)
    tstar = thr[best_idx]
    predsJ = (probs >= tstar).astype(int)
    accJ = accuracy_score(y, predsJ)
    f1J = f1_score(y, predsJ)
    
    # Brier score (MSE for probabilities)
    brier = brier_score_loss(y, probs)
    
    # Baseline Brier (always predicting base rate)
    baseline_brier = pos_rate * (1 - pos_rate)
    brier_skill = 1 - (brier / baseline_brier) if baseline_brier > 0 else 0
    
    # Confusion matrix at best threshold
    tn, fp, fn, tp = confusion_matrix(y, predsJ).ravel()
    
    return {
        'pos_rate': float(pos_rate),
        'acc@0.5': float(acc05),
        'f1@0.5': float(f105),
        'roc_auc': float(auc),
        'pr_auc': float(ap),
        'best_thr': float(tstar),
        'acc@best': float(accJ),
        'f1@best': float(f1J),
        'brier_score': float(brier),
        'brier_skill': float(brier_skill),
        'tpr@best': float(tpr[best_idx]),
        'fpr@best': float(fpr[best_idx]),
        'tp': int(tp), 'fp': int(fp), 
        'tn': int(tn), 'fn': int(fn),
        'mean_prob': float(probs.mean()),
        'std_prob': float(probs.std())
    }


def compute_calibration(probs: np.ndarray, targets: np.ndarray, 
                        n_bins: int = 10) -> Dict:
    '''
    Compute calibration metrics including ECE
    
    Args:
        probs: Predicted probabilities
        targets: True labels
        n_bins: Number of calibration bins
    
    Returns:
        Calibration metrics
    '''
    probs = np.asarray(probs).ravel()
    y = np.asarray(targets).ravel()
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    mce = 0
    bin_info = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Accuracy in bin (fraction of positives)
            accuracy_in_bin = y[in_bin].mean()
            # Average confidence in bin
            avg_confidence_in_bin = probs[in_bin].mean()
            # Calibration error for this bin
            bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
            
            ece += prop_in_bin * bin_error
            mce = max(mce, bin_error)
            
            bin_info.append({
                'range': f'({bin_lower:.2f}, {bin_upper:.2f}]',
                'count': int(in_bin.sum()),
                'accuracy': float(accuracy_in_bin),
                'confidence': float(avg_confidence_in_bin),
                'error': float(bin_error)
            })
    
    return {
        'ece': float(ece),
        'mce': float(mce),
        'n_bins_used': len(bin_info),
        'bins': bin_info
    }


def plot_binary_metrics(probs: np.ndarray, targets: np.ndarray, 
                        save_path: Optional[str] = None):
    '''
    Plot ROC, PR curves, and calibration plot
    '''
    probs = np.asarray(probs).ravel()
    y = np.asarray(targets).astype(int).ravel()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y, probs)
    auc = roc_auc_score(y, probs)
    axes[0, 0].plot(fpr, tpr, label=f'ROC (AUC = {auc:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PR Curve
    precision, recall, _ = precision_recall_curve(y, probs)
    ap = average_precision_score(y, probs)
    axes[0, 1].plot(recall, precision, label=f'PR (AP = {ap:.3f})')
    axes[0, 1].axhline(y.mean(), color='k', linestyle='--', label=f'Baseline = {y.mean():.3f}')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Calibration Plot
    cal_data = compute_calibration(probs, y, n_bins=10)
    if cal_data['bins']:
        bin_confs = [b['confidence'] for b in cal_data['bins']]
        bin_accs = [b['accuracy'] for b in cal_data['bins']]
        axes[1, 0].plot(bin_confs, bin_accs, 'o-', label=f'ECE = {cal_data["ece"]:.3f}')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        axes[1, 0].set_xlabel('Mean Predicted Probability')
        axes[1, 0].set_ylabel('Fraction of Positives')
        axes[1, 0].set_title('Calibration Plot')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Probability Distribution
    axes[1, 1].hist(probs[y == 0], bins=30, alpha=0.5, label='Class 0', density=True)
    axes[1, 1].hist(probs[y == 1], bins=30, alpha=0.5, label='Class 1', density=True)
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Probability Distributions by Class')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    return fig


def evaluate_checkpoint(checkpoint_path: str, dataloader, device='cuda'):
    '''
    Evaluate a saved checkpoint with binary metrics
    '''
    from bef_eeg.pipeline import BEF_EEG
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize model (adjust parameters as needed)
    model = BEF_EEG(
        in_chans=129,
        sfreq=100,
        n_paths=64,
        K=16
    ).to(device)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Collect predictions
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['eeg'].to(device)
            targets = batch['target'].to(device)
            
            outputs = model(inputs)
            probs = torch.sigmoid(outputs['logits']).cpu().numpy()
            
            all_probs.extend(probs)
            all_targets.extend(targets.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Compute metrics
    metrics = eval_binary(all_probs, all_targets)
    cal_metrics = compute_calibration(all_probs, all_targets)
    
    return {**metrics, **cal_metrics}, all_probs, all_targets


if __name__ == '__main__':
    # Test with dummy data
    np.random.seed(42)
    n = 1000
    
    # Simulate somewhat calibrated predictions
    true_probs = np.random.beta(2, 3, n)  # Skewed toward 0
    targets = (np.random.random(n) < true_probs).astype(int)
    probs = true_probs + np.random.normal(0, 0.1, n)
    probs = np.clip(probs, 0, 1)
    
    print('Testing binary evaluation functions...')
    print('=' * 50)
    
    metrics = eval_binary(probs, targets)
    print('\nBinary Metrics:')
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f'  {k:15s}: {v:.4f}')
        else:
            print(f'  {k:15s}: {v}')
    
    print('\nCalibration Metrics:')
    cal = compute_calibration(probs, targets)
    print(f'  ECE: {cal["ece"]:.4f}')
    print(f'  MCE: {cal["mce"]:.4f}')
    print(f'  Bins used: {cal["n_bins_used"]}')
    
    # Save plot
    fig = plot_binary_metrics(probs, targets, save_path='binary_metrics_test.png')
    print('\nPlot saved to binary_metrics_test.png')
