# BEF Pipeline for EEG Decoding

## Mathematical Framework

### BICEP: Stochastic Path Generation

BICEP generates N stochastic sample paths from an input EEG signal using the Ornstein-Uhlenbeck SDE:

```
dX_t = θ(μ - X_t)dt + σdW_t
```

Where θ controls mean reversion rate, μ is the long-term mean, and σ is volatility. Implementation uses Euler-Maruyama discretization with timestep dt:

```
X_{t+1} = X_t + θ(μ - X_t)dt + σ√dt · ε_t,  ε_t ~ N(0,1)
```

For EEG, we add jump diffusion to model event-related potentials. The model generates N=64 paths per input trial to quantify trajectory uncertainty.

### ENN: Multi-State Entanglement

ENN maintains K=16 latent states per neuron instead of scalar activations. State evolution follows:

```
h_{t+1} = tanh(W_x·x_t + E·h_t - λ·h_t + b)
```

Where E = L·L^T ensures positive semi-definite entanglement (L is learned Cholesky factor), and λ is a decay rate. The collapse mechanism uses attention:

```
α = softmax(W_g·h_T / τ)
z = Σ_k α_k·h_{T,k}
```

This delays state reduction until the final layer, preserving multiple hypotheses throughout processing.

### Fusion Alpha: Graph-Based Aggregation

Constructs a graph G=(V,E) where nodes represent EEG channels and edges encode relationships. Edge weights combine correlation and spatial proximity:

```
A_{ij} = exp(cos(x_i, x_j) / T) · exp(-d_{ij}^2 / 2σ^2)
```

Message passing follows graph convolution:

```
H^{(l+1)} = σ(D^{-1/2}AD^{-1/2}H^{(l)}W^{(l)})
```

After L layers, nodes are aggregated via attention-weighted pooling for final prediction.

## Pipeline Architecture

```
Input: EEG [B, 129, 200]
    |
    v
BICEP: N=64 stochastic paths via Ornstein-Uhlenbeck SDE
    |
    v
ENN: K=16 entangled states with attention collapse
    |
    v
Fusion Alpha: Graph convolution over channel nodes
    |
    v
Output: Predictions [B, 1] + Uncertainty [B, 1]
```

## Implementation Details

### Files
- `bicep_eeg.py`: SDE simulator with Euler-Maruyama integration
- `enn.py`: Entangled RNN with PSD constraint via Cholesky factorization
- `fusion_alpha.py`: Graph neural network with normalized Laplacian propagation
- `pipeline.py`: End-to-end BEF orchestration
- `train.py`: Three-stage training (pretrain, fusion fine-tune, full fine-tune)
- `config.yaml`: Hyperparameters (N_paths=64, K=16, GNN layers=3)
- `model.py`, `submission.py`: Codabench submission interfaces

### Training Protocol

1. Pretrain ENN with contrastive loss (10 epochs)
2. Fine-tune Fusion Alpha with frozen BICEP/ENN (20 epochs)
3. Fine-tune full pipeline with frozen BICEP (50 epochs)

Learning rates decay from 1e-3 to 1e-4 across stages.

### Usage

```python
from pipeline import BEF_EEG

model = BEF_EEG(in_chans=129, sfreq=100, n_paths=64, K=16)
outputs = model(eeg_data)  # Returns dict with predictions and uncertainties
```

For training:
```bash
python train.py
```

## Requirements

- PyTorch >= 2.0
- NumPy, SciPy, scikit-learn
- YAML, tqdm
- Optional: wandb for logging

## Performance Metrics

Target performance on EEG Foundation Challenge:
- MAE: 85-105 ms
- R²: 0.40-0.55
- AUC: 0.82-0.88

These targets assume proper pretraining and calibration.