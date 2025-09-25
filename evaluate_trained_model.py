'''
Evaluate the trained BEF model with proper binary metrics
'''

import torch
import numpy as np
import yaml

from binary_evaluation import eval_binary, compute_calibration, plot_binary_metrics
from bef_eeg.pipeline import PretrainableBEF
from hbn_dataset_loader import create_hbn_dataloaders


def evaluate_model(checkpoint_path='bef_eeg/weights_challenge_1.pt'):
    '''Evaluate trained model with binary metrics'''
    
    print(f'Loading checkpoint from {checkpoint_path}...')
    
    # Load config
    with open('optimized_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PretrainableBEF(
        in_chans=config['in_chans'],
        sfreq=config['sfreq'],
        n_paths=config['bicep']['n_paths'],
        K=config['enn']['K'],
        gnn_hidden=config['fusion']['gnn_hid'],
        gnn_layers=config['fusion']['layers'],
        dropout=config['fusion']['dropout'],
        output_dim=1
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print('Model loaded successfully!')
    
    # Create dataloaders
    print('Loading data...')
    train_loader, val_loader, test_loader = create_hbn_dataloaders(
        config,
        tasks=["RestingState", "DespicableMe"],
    )
    
    # Evaluate on each split
    for split_name, dataloader in [('val', val_loader), ('test', test_loader)]:
        print(f'\n=== Evaluating {split_name.upper()} set ===')
        
        # Collect predictions
        all_probs = []
        all_targets = []
        
        with torch.no_grad():
            for eeg, label in dataloader:
                inputs = eeg.to(device)
                targets = (label == 1).long()

                outputs = model(inputs)
                logits = outputs['prediction'].squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()

                all_probs.extend(probs)
                all_targets.extend(targets.numpy())
        
        all_probs = np.array(all_probs)
        all_targets = np.array(all_targets)
        
        # Compute metrics
        print(f'Computing metrics for {len(all_probs)} samples...')
        metrics = eval_binary(all_probs, all_targets)
        cal_metrics = compute_calibration(all_probs, all_targets)
        
        # Print results
        print(f'\n{split_name.upper()} Results:')
        print('-' * 50)
        
        # Key metrics comparison with baseline
        baseline_brier = metrics['pos_rate'] * (1 - metrics['pos_rate'])
        print(f'\nClass Balance:')
        print(f'  Positive rate: {metrics["pos_rate"]:.3f}')
        print(f'  Baseline Brier: {baseline_brier:.4f}')
        
        print(f'\nProbability Metrics:')
        print(f'  Brier Score: {metrics["brier_score"]:.4f} (lower is better)')
        print(f'  Brier Skill: {metrics["brier_skill"]:.3f} ({metrics["brier_skill"]*100:.1f}% better than baseline)')
        print(f'  Mean Prob: {metrics["mean_prob"]:.3f} (should match pos_rate if calibrated)')
        print(f'  Std Prob: {metrics["std_prob"]:.3f}')
        
        print(f'\nClassification Metrics:')
        print(f'  ROC-AUC: {metrics["roc_auc"]:.4f} (random=0.5)')
        print(f'  PR-AUC: {metrics["pr_auc"]:.4f} (baseline={metrics["pos_rate"]:.3f})')
        
        print(f'\nAt Default Threshold (0.5):')
        print(f'  Accuracy: {metrics["acc@0.5"]:.4f}')
        print(f'  F1 Score: {metrics["f1@0.5"]:.4f}')
        
        print(f'\nAt Optimal Threshold ({metrics["best_thr"]:.3f}):')
        print(f'  Accuracy: {metrics["acc@best"]:.4f} (+{(metrics["acc@best"]-metrics["acc@0.5"])*100:.1f}%)')
        print(f'  F1 Score: {metrics["f1@best"]:.4f} (+{(metrics["f1@best"]-metrics["f1@0.5"])*100:.1f}%)')
        print(f'  TPR: {metrics["tpr@best"]:.3f}, FPR: {metrics["fpr@best"]:.3f}')
        
        print(f'\nCalibration:')
        print(f'  ECE: {cal_metrics["ece"]:.4f} (lower is better)')
        print(f'  MCE: {cal_metrics["mce"]:.4f} (lower is better)')
        
        print(f'\nConfusion Matrix @ Optimal Threshold:')
        print(f'              Predicted')
        print(f'              Neg   Pos')
        print(f'  Actual Neg  {metrics["tn"]:4d}  {metrics["fp"]:4d}')
        print(f'  Actual Pos  {metrics["fn"]:4d}  {metrics["tp"]:4d}')
        
        # Save plots
        plot_path = f'{split_name}_binary_metrics.png'
        plot_binary_metrics(all_probs, all_targets, save_path=plot_path)
        print(f'\nPlots saved to {plot_path}')
    
    print('\n' + '='*50)
    print('Evaluation complete!')
    
    return metrics


if __name__ == '__main__':
    evaluate_model()
