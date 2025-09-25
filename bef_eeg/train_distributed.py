"""
BEF Training Script with RunPod Optimizations
- AMP (mixed precision) support
- S3 streaming
- OOM protection
- Time-based checkpointing
- Job-level sharding
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import yaml
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

# Import existing modules
try:
    from .pipeline import BEF_EEG
except ImportError:
    from pipeline import BEF_EEG
try:
    from .utils_io import load_eeg_data
except ImportError:
    from utils_io import load_eeg_data

try:
    from .train import BEFTrainer
except ImportError:
    from train import BEFTrainer

try:
    from .hbn_dataloader import HBNDataset, create_hbn_dataloader
except ImportError:
    from hbn_dataloader import HBNDataset, create_hbn_dataloader

try:
    from .training_utils import (
        setup_logging,
        auto_reduce_batch_size,
        time_based_checkpoint,
        throughput_probe,
        load_config_with_env
    )
except ImportError:
    from training_utils import (
        setup_logging,
        auto_reduce_batch_size,
        time_based_checkpoint,
        throughput_probe,
        load_config_with_env
    )
    setup_logging,
    auto_reduce_batch_size,
    time_based_checkpoint,
    throughput_probe,
    load_config_with_env
)


class DistributedBEFTrainer(BEFTrainer):
    """Extended trainer with RunPod optimizations"""

    def __init__(
        self,
        model: BEF_EEG,
        config: Dict,
        device: str = "cuda",
        use_wandb: bool = False,
        shard_id: int = 0,
        num_shards: int = 1,
        log_dir: str = "/workspace/logs",
        checkpoint_dir: str = "/workspace/checkpoints",
        use_amp: bool = True,
        amp_dtype: str = "bfloat16"
    ):
        super().__init__(model, config, device, use_wandb)

        self.shard_id = shard_id
        self.num_shards = num_shards
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.use_amp = use_amp

        # Setup AMP
        self.scaler = GradScaler(enabled=use_amp)
        if amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float16

        # Time tracking
        self.start_time = time.time()
        self.next_checkpoint_time = time.time() + config.get('checkpoint_interval_min', 45) * 60

        # OOM handling
        self.current_batch_size = config['dataloader']['batch_size']
        self.grad_accum_steps = config['training'].get('grad_accum_steps', 2)
        self.min_batch_size = config['training'].get('min_batch_size', 1)

        # Logging
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_step(self, batch: Dict, optimizer: optim.Optimizer) -> Dict:
        """Single training step with AMP and OOM protection"""
        try:
            # Forward pass with mixed precision
            with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(batch['eeg'])
                loss = outputs['loss']

                # Scale loss for gradient accumulation
                loss = loss / self.grad_accum_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            return {'loss': loss.item() * self.grad_accum_steps}

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Handle OOM
                torch.cuda.empty_cache()
                return self.handle_oom(batch, optimizer)
            else:
                raise

    def handle_oom(self, batch: Dict, optimizer: optim.Optimizer) -> Dict:
        """Handle OOM by reducing batch size"""
        print(f"OOM detected! Reducing batch size from {self.current_batch_size}")

        if self.current_batch_size <= self.min_batch_size:
            print(f"Already at minimum batch size {self.min_batch_size}, skipping batch")
            return {'loss': float('nan')}

        # Reduce batch size and increase accumulation
        self.current_batch_size = max(self.current_batch_size // 2, self.min_batch_size)
        self.grad_accum_steps *= 2

        print(f"New batch size: {self.current_batch_size}, grad_accum: {self.grad_accum_steps}")

        # Clear gradients
        optimizer.zero_grad(set_to_none=True)

        # Return NaN loss to skip this step
        return {'loss': float('nan')}

    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        epoch: int,
        max_time_min: Optional[float] = None
    ) -> Dict:
        """Train for one epoch with time cap"""
        self.model.train()
        epoch_start = time.time()
        max_time_sec = max_time_min * 60 if max_time_min else float('inf')

        losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(pbar):
            # Check time limit
            if time.time() - epoch_start > max_time_sec:
                print(f"Time limit reached ({max_time_min} min), stopping epoch")
                break

            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            # Training step
            metrics = self.train_step(batch, optimizer)

            if not np.isnan(metrics['loss']):
                losses.append(metrics['loss'])

            # Gradient accumulation
            if (step + 1) % self.grad_accum_steps == 0:
                # Gradient clipping
                if self.config['optimization'].get('grad_clip_norm'):
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['optimization']['grad_clip_norm']
                    )

                # Optimizer step
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Update progress bar
            if losses:
                pbar.set_postfix({'loss': np.mean(losses[-100:])})

            # Time-based checkpointing
            if time.time() >= self.next_checkpoint_time:
                self.save_checkpoint(epoch, step, optimizer)
                self.next_checkpoint_time = time.time() + self.config.get('checkpoint_interval_min', 45) * 60

        return {
            'loss': np.mean(losses) if losses else float('nan'),
            'epoch_time': time.time() - epoch_start,
            'steps': len(losses)
        }

    def save_checkpoint(self, epoch: int, step: int, optimizer: optim.Optimizer):
        """Save checkpoint with metadata"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config,
            'metrics': self.best_metrics,
            'batch_size': self.current_batch_size,
            'grad_accum_steps': self.grad_accum_steps,
            'time_elapsed': time.time() - self.start_time
        }

        # Save locally
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch{epoch}_step{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # Also save as 'last.pt' for easy resume
        torch.save(checkpoint, self.checkpoint_dir / "last.pt")

        # Upload to S3 if configured
        if self.config.get('artifacts', {}).get('upload_to_s3'):
            self.upload_to_s3(checkpoint_path)

    def upload_to_s3(self, file_path: Path):
        """Upload file to S3"""
        s3_bucket = self.config['artifacts'].get('s3_bucket')
        s3_prefix = self.config['artifacts'].get('s3_prefix', 'hbn-daypush')

        if not s3_bucket:
            return

        import boto3
        s3 = boto3.client('s3')

        s3_key = f"{s3_prefix}/{self.shard_id}/{file_path.name}"
        try:
            s3.upload_file(str(file_path), s3_bucket, s3_key)
            print(f"Uploaded to s3://{s3_bucket}/{s3_key}")
        except Exception as e:
            print(f"S3 upload failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="BEF-EEG Training with RunPod Optimizations")

    # Core arguments
    parser.add_argument("--config", type=str, default="runpod_config.yaml")
    parser.add_argument("--release", type=str, required=True, help="HBN release (e.g., R01)")
    parser.add_argument("--use-mini", type=str, default="false", choices=["true", "false"])

    # Time management
    parser.add_argument("--time-cap-min", type=int, default=150, help="Max time per dataset in minutes")
    parser.add_argument("--checkpoint-interval-min", type=int, default=45)

    # Paths
    parser.add_argument("--checkpoint-dir", type=str, default="/workspace/checkpoints")
    parser.add_argument("--log-dir", type=str, default="/workspace/logs")
    parser.add_argument("--log-file", type=str, default="train.log")

    # S3 configuration
    parser.add_argument("--s3-bucket", type=str, default="")
    parser.add_argument("--s3-prefix", type=str, default="hbn-daypush")

    # Probe mode
    parser.add_argument("--probe-only", action="store_true", help="Run throughput probe only")
    parser.add_argument("--probe-steps", type=int, default=200)

    # Resume training
    parser.add_argument("--resume-from", type=str, default="", help="Path to checkpoint")

    args = parser.parse_args()

    # Load configuration
    config = load_config_with_env(args.config)

    # Override with command line args
    config['data']['use_mini'] = args.use_mini == "true"
    config['training']['max_time_per_dataset_min'] = args.time_cap_min
    config['training']['checkpoint_interval_min'] = args.checkpoint_interval_min

    if args.s3_bucket:
        config['artifacts']['s3_bucket'] = args.s3_bucket
        config['artifacts']['s3_prefix'] = args.s3_prefix

    # Setup logging
    logger = setup_logging(args.log_dir, args.log_file)
    logger.info(f"Starting training for release {args.release}")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model
    model = BEF_EEG(
        in_chans=config['model']['in_chans'],
        sfreq=config['model']['sfreq'],
        n_paths=config['model']['n_paths'],
        K=config['model']['K']
    ).to(device)

    # Load checkpoint if resuming
    start_epoch = 0
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        logger.info(f"Resumed from checkpoint: {args.resume_from}, epoch {start_epoch}")

    # Probe mode
    if args.probe_only:
        logger.info("Running throughput probe...")

        # Create dummy dataloader
        dummy_dataset = HBNDataset(
            release=args.release,
            use_mini=config['data']['use_mini'],
            config=config['data']
        )

        probe_loader = create_hbn_dataloader(
            dummy_dataset,
            batch_size=config['dataloader']['batch_size'],
            num_workers=2,  # Reduced for probe
            shuffle=False
        )

        # Run probe
        steps_per_sec = throughput_probe(
            model, probe_loader, device,
            num_steps=args.probe_steps,
            use_amp=config['training']['amp_enabled'],
            amp_dtype=config['training']['amp_dtype']
        )

        logger.info(f"Steps/sec: {steps_per_sec:.2f}")
        print(f"Steps/sec: {steps_per_sec:.2f}")
        return

    # Create dataset and dataloader
    logger.info(f"Loading HBN dataset: {args.release} (mini={config['data']['use_mini']})")

    dataset = HBNDataset(
        release=args.release,
        use_mini=config['data']['use_mini'],
        config=config['data']
    )

    dataloader = create_hbn_dataloader(
        dataset,
        batch_size=config['dataloader']['batch_size'],
        num_workers=config['dataloader']['num_workers'],
        prefetch_factor=config['dataloader']['prefetch_factor'],
        pin_memory=config['dataloader']['pin_memory'],
        persistent_workers=config['dataloader']['persistent_workers'],
        shuffle=config['dataloader']['shuffle']
    )

    # Create trainer
    trainer = DistributedBEFTrainer(
        model=model,
        config=config,
        device=device,
        use_wandb=False,  # Disabled for RunPod
        shard_id=int(os.environ.get('SHARD_ID', 0)),
        num_shards=int(os.environ.get('NUM_SHARDS', 1)),
        log_dir=args.log_dir,
        checkpoint_dir=f"{args.checkpoint_dir}/{args.release}",
        use_amp=config['training']['amp_enabled'],
        amp_dtype=config['training']['amp_dtype']
    )

    # Setup optimizer
    trainer.setup_optimizers("finetune_full")
    optimizer = trainer.optimizers['full']

    # Training loop
    logger.info("Starting training loop")
    train_start = time.time()
    max_time_sec = args.time_cap_min * 60

    for epoch in range(start_epoch, config['training']['epochs_finetune']):
        # Check time limit
        if time.time() - train_start > max_time_sec:
            logger.info(f"Time limit reached ({args.time_cap_min} min), stopping training")
            break

        # Train epoch
        epoch_metrics = trainer.train_epoch(
            dataloader,
            optimizer,
            epoch,
            max_time_min=min(
                args.time_cap_min - (time.time() - train_start) / 60,
                30  # Max 30 min per epoch
            )
        )

        logger.info(f"Epoch {epoch}: loss={epoch_metrics['loss']:.4f}, "
                   f"time={epoch_metrics['epoch_time']:.1f}s, "
                   f"steps={epoch_metrics['steps']}")

        # Update scheduler if exists
        if 'full' in trainer.schedulers:
            trainer.schedulers['full'].step()

    # Final checkpoint
    trainer.save_checkpoint(epoch, 0, optimizer)

    # Summary
    total_time = time.time() - train_start
    logger.info(f"Training complete for {args.release}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Final metrics: {trainer.best_metrics}")

    # Save summary
    summary = {
        'release': args.release,
        'use_mini': config['data']['use_mini'],
        'total_time_min': total_time / 60,
        'epochs_completed': epoch - start_epoch,
        'final_batch_size': trainer.current_batch_size,
        'final_grad_accum': trainer.grad_accum_steps,
        'best_metrics': trainer.best_metrics
    }

    summary_path = Path(args.log_dir) / f"{args.release}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()