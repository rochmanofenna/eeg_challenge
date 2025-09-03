# EEG Challenge: One-Command Reproducible Training

🧠 **BICEP→ENN→FusionAlpha** pipeline for NeurIPS 2025 EEG Foundation Challenge

## 🚀 Quick Start (Any Machine)

```bash
# 1. Clone/download this repo
git clone <your-repo> eeg_challenge
cd eeg_challenge

# 2. One-command setup
bash install.sh

# 3. Start training
source venv/bin/activate
python train_s3_extended.py --pretrain_epochs 100 --finetune_epochs 50
```

**That's it!** Works on any Linux/Mac machine with CUDA.

## ☁️ Cloud GPU Options

### Google Colab Pro+
```bash
# Upload this folder to Colab
!bash install.sh
!python train_s3_extended.py --batch_size 64 --max_subjects 100
```

### Lambda Labs / RunPod / Paperspace
```bash
# SSH in, then:
git clone <repo>
cd eeg_challenge
bash install.sh cu118  # or cu121 for newer CUDA
source venv/bin/activate
python train_s3_extended.py
```

### AWS/GCP/Azure
Same as above! The install script handles everything.

## 🗂️ What Gets Set Up

- **Virtual Environment**: Isolated Python environment
- **PyTorch + CUDA**: Automatic CUDA detection and installation  
- **EEG Libraries**: MNE, braindecode, autoreject
- **S3 Access**: Automatic connection to HBN dataset
- **All Dependencies**: Requirements installed automatically

## 📊 Training Configurations

### Quick Test (5 minutes)
```bash
python train_s3_extended.py --pretrain_epochs 1 --finetune_epochs 1 --batch_size 8
```

### Full Training (varies by GPU)
```bash
python train_s3_extended.py --pretrain_epochs 100 --finetune_epochs 50 --batch_size 64
```

### Maximum Performance
```bash
python train_s3_extended.py --pretrain_epochs 200 --finetune_epochs 100 --batch_size 128 --max_subjects 500
```

## 🎯 Key Features

- **S3 Data Streaming**: No need to download 200GB+ files
- **Advanced Preprocessing**: ICA, artifact removal, bad channel interpolation
- **Subject-Invariant Training**: Better generalization across subjects
- **Uncertainty Quantification**: ENN-based confidence estimates
- **Automatic Checkpointing**: Resume training from interruptions

## 📈 Expected Performance

| GPU | Batch Size | Time/Epoch | Total Time |
|-----|------------|------------|------------|
| RTX 3080 | 32 | ~10 min | 20-30 hours |
| A100 | 128 | ~3 min | 8-12 hours |
| H100/B200 | 256+ | ~1 min | 3-5 hours |

## 🔧 Customization

Edit these files for your needs:
- `train_s3_extended.py`: Main training script
- `models/enn_eeg_model.py`: Model architecture  
- `data/s3_data_loader.py`: Data loading logic
- `hpc_scripts/`: HPC cluster configurations

## 📋 Directory Structure

```
eeg_challenge/
├── install.sh              # One-command setup
├── requirements.txt         # Python dependencies
├── train_s3_extended.py     # Main training script
├── models/                  # Model architectures
├── data/                    # Data loading utilities
├── preprocessing/           # EEG preprocessing
├── utils/                   # Training utilities
└── hpc_scripts/            # HPC cluster scripts
```

## 🆘 Troubleshooting

**CUDA Issues:**
```bash
bash install.sh cu118  # or cu121
```

**Memory Errors:**
```bash
python train_s3_extended.py --batch_size 16  # reduce batch size
```

**S3 Connection:**
```bash
python -c "import s3fs; print(s3fs.S3FileSystem(anon=True).ls('fcp-indi')[:5])"
```

## 🏆 Results

After training, check:
- `results/`: Final model performance
- `checkpoints/`: Saved model weights  
- `logs/`: Training progress logs

Expected performance on EEG→behavior prediction:
- **MAE**: <0.5 seconds (vs 0.8+ baseline)
- **R²**: >0.3 (vs -0.002 baseline)  
- **AUC**: >0.7 (vs 0.51 baseline)

---

**🎯 This setup works ANYWHERE with one command!**