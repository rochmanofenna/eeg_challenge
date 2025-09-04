# BEF Pipeline for EEG Foundation Challenge

## Overview

Complete implementation of the **BICEP → ENN → Fusion Alpha** (BEF) pipeline optimized for the 2025 EEG Foundation Challenge. This implementation addresses the weaknesses in the basic version by restoring the full theoretical capabilities of each component.

## Key Improvements from Basic Implementation

### BICEP (Brownian Iterative Continuous Estimation & Prediction)
- **Before**: Single-path or mock simulation, no true stochastic modeling
- **Now**: Full multi-future Ornstein-Uhlenbeck SDE with N=64 paths
- **Features**:
  - EEG-specific SDE modeling (oscillatory dynamics for SSVEP)
  - Event-related jumps for ERP simulation
  - Antithetic sampling for variance reduction
  - Learned adaptive parameters

### ENN (Entangled Neural Network)
- **Before**: Simple probabilistic head with mean/variance output
- **Now**: True K=16 state entanglement with delayed collapse
- **Features**:
  - PSD entanglement matrix E = L·L^T
  - Recurrent processing with multiple hypothesis tracking
  - Attention-based collapse mechanism
  - Multi-scale temporal processing option

### Fusion Alpha (Graph-Based Fusion)
- **Before**: Fixed KNN smoothing, no learning
- **Now**: Trainable GNN with contradiction resolution
- **Features**:
  - Graph Attention Network layers
  - Dynamic edge weights based on correlation/spatial proximity
  - MC Dropout for calibrated uncertainty
  - Hierarchical fusion option (channel → region → global)

## Architecture

```
EEG Input [B, 129, 200]
    ↓
BICEP: Simulate N=64 stochastic paths
    ↓
ENN: Process through K=16 entangled states
    ↓
Fusion Alpha: GNN contradiction resolution
    ↓
Output: Predictions + Calibrated Uncertainty
```

## Training Strategy

### Stage 1: Pretraining (10 epochs)
- Contrastive learning on ENN representations
- Learn general EEG features without labels

### Stage 2: Fusion Fine-tuning (20 epochs)
- Freeze BICEP & ENN
- Train only Fusion Alpha on target task
- Quick adaptation to new decision boundaries

### Stage 3: Full Fine-tuning (50 epochs)
- Unfreeze ENN (keep BICEP frozen)
- Joint training with cosine annealing
- Final calibration and optimization

## File Structure

```
models/
├── bicep_eeg.py        # Stochastic multi-future simulator
├── enn.py              # Entangled neural network
├── fusion_alpha.py     # Graph neural network fusion
└── bef_pipeline.py     # Unified BEF implementation

train_bef.py            # Training infrastructure
model.py                # Codabench submission (variant A)
submission_bef.py       # Codabench submission (variant B)
config_bef.yaml         # Configuration file
```

## Usage

### Training

```bash
# Full training pipeline
python train_bef.py

# Custom configuration
python train_bef.py --config my_config.yaml
```

### Inference

```python
from model import Model

# Initialize (auto-loads weights)
model = Model()

# Predict
eeg_data = torch.randn(32, 129, 200)  # [batch, channels, time]
predictions = model(eeg_data)
```

### Submission

Both submission formats are provided:

1. **model.py**: Uses no-args constructor
2. **submission_bef.py**: Uses SFREQ/DEVICE constructor

## Performance Targets

Based on the theoretical analysis and improvements:

| Metric | Conservative | Optimized BEF | Stretch Goal |
|--------|-------------|---------------|--------------|
| R² (RT) | 0.25-0.35 | 0.40-0.55 | 0.60-0.70 |
| MAE (ms) | 100-120 | 85-105 | 75-90 |
| AUC | 0.74-0.80 | 0.82-0.88 | 0.89-0.93 |

## Key Features

1. **Uncertainty Quantification**
   - Epistemic uncertainty from MC Dropout
   - Aleatoric uncertainty from ENN entropy
   - Calibrated total uncertainty

2. **Transfer Learning**
   - Pretrain on passive tasks
   - Fine-tune on active tasks
   - Cross-subject graph transfer

3. **Contradiction Resolution**
   - Explicit detection of conflicting signals
   - Graph-based reconciliation
   - Attention-weighted fusion

4. **Computational Efficiency**
   - Reduced paths (32) for inference
   - Optional hierarchical processing
   - GPU-optimized operations

## Requirements

```
torch>=2.0.0
numpy
scipy
scikit-learn
tqdm
wandb (optional)
pyyaml
```

## Notes

- All components follow the BEF axioms:
  - A1: Multiplicity (BICEP multi-futures)
  - A2: Delay-of-collapse (ENN entanglement)
  - A3: Contradiction resolution (Fusion Alpha)
  - A4: Calibrated uncertainty (full pipeline)

- The implementation is modular - each component can be tested/replaced independently

- Cryptographic extensions are not implemented but the architecture supports future integration

## Citation

If using this implementation, please acknowledge the BEF framework and EEG Foundation Challenge.