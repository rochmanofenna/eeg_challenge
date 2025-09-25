# BEF EEG Challenge Package

## Overview
- **Model**: BICEP → ENN → Fusion Alpha ("BEF") pipeline for HBN EEG decoding.
- **Targets**: Challenge 1 (binary cross-task success) and Challenge 2 (P-factor regression).
- **Current best**: Multi-GPU training on three 4090s achieved ≈0.3338 MAE / 0.1968 MSE (R²≈0.089) on the P-factor task.

## Repository Layout
- `bef_eeg/`
  - Core modules (`bicep_eeg.py`, `enn.py`, `fusion_alpha.py`, `pipeline.py`).
  - Training utilities (`train.py`, `train_distributed.py`, `training_utils.py`, `utils_io.py`).
  - Inference entry points (`submission.py`, `model.py`, `test_submission.py`).
  - Saved weights (`weights_challenge_1.pt`, `weights_challenge_2.pt`).
  - Example numpy splits under `data/`.
- `binary_evaluation.py` – post-training classification metrics & calibration helpers.
- `download_hbn.sh` – grab the downsampled HBN (BDF/BIDS) releases from AWS S3.
- `evaluate.py`, `evaluate_model_fixed.py`, `evaluate_trained_model.py` – sanity checks for metrics / loaders.
- `multi_gpu_train.py`, `TRAIN_FINAL.py`, `RUN_TRAINING.sh` – orchestration scripts for multi-GPU or legacy experiments.
- `hbn_dataset_loader.py`, `hbn_multi_release_loader.py`, `real_hbn_loader.py` – dataset utilities (local splits + S3 streaming).
- Notes & logs (`BINARY_IMPROVEMENTS_SUMMARY.md`, `GPU_STATUS.md`, `hbn_dataset_summary.json`, `hbn_training_results.json`).

## Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
The list covers PyTorch + data tooling (`mne`, `boto3`) used across training and S3 ingestion.

## Inference
```python
from bef_eeg.submission import Submission
import torch

model = Submission(SFREQ=100, DEVICE='cpu')
# Dummy EEG batch: [batch, 129 channels, 200 samples]
x = torch.randn(2, 129, 200)
outputs = model.get_model_challenge_2()(x)
print(outputs['prediction'])
```
- Challenge‑1 wrapper: `Submission.get_model_challenge_1()` → dict with `rt` & `success` tensors.
- Challenge‑2 wrapper: `Submission.get_model_challenge_2()` → regression head for P-factor.
- Models auto-load `bef_eeg/weights_challenge_1.pt` and `bef_eeg/weights_challenge_2.pt`.

## Evaluation & Training (optional)
- **Binary metrics**: `python evaluate_model_fixed.py` (real HBN validation windows) or `python evaluate_trained_model.py` (full loader).
- **Regression sanity**: `python evaluate.py` (uses `bef_eeg/train.py` evaluation loop).
- **Multi-GPU finetune**: `python multi_gpu_train.py --config bef_eeg/config.yaml` (expects 1–3 GPUs, streams from S3 if available).
- **Single-GPU / staged training**: `python -m bef_eeg.train` (see `bef_eeg/config.yaml` for hyperparameters).

## Preparing a Codabench Submission
1. Ensure `bef_eeg/weights_challenge_1.pt` and `bef_eeg/weights_challenge_2.pt` are the desired checkpoints.
2. Copy the minimal inference payload into a clean directory:
   - `bef_eeg/submission.py`
   - `bef_eeg/model.py`
   - `bef_eeg/pipeline.py`, `bef_eeg/bicep_eeg.py`, `bef_eeg/enn.py`, `bef_eeg/fusion_alpha.py`, `bef_eeg/utils_io.py`, `bef_eeg/training_utils.py`, `bef_eeg/hbn_dataloader.py`
   - `bef_eeg/config.yaml`
   - `bef_eeg/weights_challenge_1.pt`, `bef_eeg/weights_challenge_2.pt`
   - `requirements.txt` (or a trimmed copy listing only runtime deps)
3. Zip the directory (no training scripts or data) and upload via the EEG 2025 submission portal.

## Notes
- All imports now use the `bef_eeg` package namespace; running scripts from the repo root works without manual `sys.path` tweaks.
- Logs, scratch scripts, and version-control artefacts have been removed to keep the tree lightweight.
- For reproducibility, keep `config.yaml` synced with the checkpoints you plan to ship.
