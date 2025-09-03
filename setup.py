#!/usr/bin/env python3
"""
Setup script for EEG Challenge training pipeline
Makes the entire project reproducible with one command
"""

from setuptools import setup, find_packages

setup(
    name="eeg-challenge",
    version="1.0.0",
    description="BICEP→ENN→FusionAlpha pipeline for NeurIPS 2025 EEG Challenge",
    author="Ryan",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision",
        "numpy==1.24.3",
        "mne>=1.5.0",
        "braindecode>=0.8.0",
        "scikit-learn",
        "scipy",
        "pandas",
        "autoreject",
        "pyriemann",
        "s3fs",
        "boto3",
        "tqdm",
        "wandb",
        "matplotlib",
        "seaborn"
    ],
    extras_require={
        "dev": ["jupyter", "ipykernel"],
        "hpc": ["mpi4py"]
    },
    entry_points={
        "console_scripts": [
            "eeg-train=train_s3_extended:main",
            "eeg-test=test_pipeline:main"
        ]
    }
)