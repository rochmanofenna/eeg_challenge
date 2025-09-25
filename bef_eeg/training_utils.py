"""
Training utilities for RunPod deployment
- Logging setup
- OOM handling
- Time-based checkpointing
- Throughput probing
- Configuration management
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import yaml

import torch
from torch.cuda.amp import autocast
import numpy as np
from tqdm import tqdm


def setup_logging(log_dir: str, log_file: str = "train.log") -> logging.Logger:
    """Setup logging with file and console output"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("BEF-Training")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_dir / log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger


def load_config_with_env(config_path: str) -> Dict:
    """Load configuration with environment variable overrides"""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Environment variable mappings
    env_mappings = {
        'SHARD_ID': lambda v: {'experiment': {'shard_id': int(v)}},
        'NUM_SHARDS': lambda v: {'experiment': {'num_shards': int(v)}},
        'RELEASES': lambda v: {'data': {'release_whitelist': v.split(',')}},
        'USE_MINI': lambda v: {'data': {'use_mini': v.lower() == 'true'}},
        'S3_BUCKET': lambda v: {'artifacts': {'s3_bucket': v}},
        'S3_PREFIX': lambda v: {'artifacts': {'s3_prefix': v}},
        'TIME_CAP_MIN': lambda v: {'training': {'max_time_per_dataset_min': int(v)}},
        'BATCH_SIZE': lambda v: {'dataloader': {'batch_size': int(v)}},
    }

    # Apply environment overrides
    for env_var, update_fn in env_mappings.items():
        if env_var in os.environ:
            value = os.environ[env_var]
            if value:
                updates = update_fn(value)
                config = deep_update(config, updates)

    return config


def deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """Deep update nested dictionaries"""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            base_dict[key] = deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def auto_reduce_batch_size(
    current_batch_size: int,
    current_accum_steps: int,
    min_batch_size: int = 1
) -> Tuple[int, int]:
    """
    Automatically reduce batch size on OOM

    Returns:
        (new_batch_size, new_accum_steps)
    """
    if current_batch_size <= min_batch_size:
        return current_batch_size, current_accum_steps

    # Halve batch size, double accumulation
    new_batch_size = max(current_batch_size // 2, min_batch_size)
    new_accum_steps = current_accum_steps * 2

    return new_batch_size, new_accum_steps


def time_based_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    checkpoint_dir: Path,
    metadata: Dict = None,
    upload_to_s3: bool = False,
    s3_bucket: str = None,
    s3_prefix: str = None
) -> Path:
    """Save checkpoint based on time interval"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch{epoch}_step{step}_{timestamp}.pt"

    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'timestamp': timestamp,
        'metadata': metadata or {}
    }

    torch.save(checkpoint, checkpoint_path)

    # Also save as 'last.pt'
    torch.save(checkpoint, checkpoint_dir / "last.pt")

    # Upload to S3 if configured
    if upload_to_s3 and s3_bucket:
        try:
            import boto3
            s3 = boto3.client('s3')
            s3_key = f"{s3_prefix}/checkpoints/{checkpoint_path.name}"
            s3.upload_file(str(checkpoint_path), s3_bucket, s3_key)
            print(f"Uploaded checkpoint to s3://{s3_bucket}/{s3_key}")
        except Exception as e:
            print(f"S3 upload failed: {e}")

    return checkpoint_path


def throughput_probe(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_steps: int = 200,
    use_amp: bool = True,
    amp_dtype: str = "bfloat16"
) -> float:
    """
    Probe throughput (steps/sec) for optimization

    Returns:
        steps_per_second
    """
    model.eval()

    # Determine AMP dtype
    if amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    # Warmup
    warmup_steps = min(10, num_steps // 10)
    data_iter = iter(dataloader)

    for _ in range(warmup_steps):
        try:
            batch = next(data_iter)
            if isinstance(batch, dict):
                batch = {k: v.to(device) if torch.is_tensor(v) else v
                        for k, v in batch.items()}
            else:
                batch = batch.to(device)

            with torch.no_grad():
                with autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
                    if isinstance(batch, dict):
                        _ = model(batch.get('eeg', batch))
                    else:
                        _ = model(batch)
        except StopIteration:
            data_iter = iter(dataloader)

    # Measure throughput
    torch.cuda.synchronize()
    start_time = time.time()

    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        if isinstance(batch, dict):
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}
        else:
            batch = batch.to(device)

        with torch.no_grad():
            with autocast(device_type='cuda', dtype=dtype, enabled=use_amp):
                if isinstance(batch, dict):
                    _ = model(batch.get('eeg', batch))
                else:
                    _ = model(batch)

    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time
    steps_per_second = num_steps / elapsed_time

    return steps_per_second


def memory_monitor(device: torch.device = None) -> Dict[str, float]:
    """Monitor GPU memory usage"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        return {}

    return {
        'allocated_gb': torch.cuda.memory_allocated(device) / 1024**3,
        'reserved_gb': torch.cuda.memory_reserved(device) / 1024**3,
        'max_allocated_gb': torch.cuda.max_memory_allocated(device) / 1024**3,
        'free_gb': (torch.cuda.get_device_properties(device).total_memory -
                   torch.cuda.memory_allocated(device)) / 1024**3
    }


def adaptive_batch_size(
    model: torch.nn.Module,
    device: torch.device,
    input_shape: Tuple[int, ...],
    initial_batch_size: int = 64,
    max_batch_size: int = 256,
    target_memory_usage: float = 0.9
) -> int:
    """Find optimal batch size for given model and GPU"""
    model.eval()

    # Binary search for optimal batch size
    low, high = 1, max_batch_size
    optimal_batch_size = initial_batch_size

    while low <= high:
        mid = (low + high) // 2

        try:
            # Test batch
            dummy_input = torch.randn(mid, *input_shape).to(device)

            with torch.no_grad():
                _ = model(dummy_input)

            # Check memory usage
            mem_stats = memory_monitor(device)
            usage_ratio = mem_stats['allocated_gb'] / (
                mem_stats['allocated_gb'] + mem_stats['free_gb']
            )

            if usage_ratio <= target_memory_usage:
                optimal_batch_size = mid
                low = mid + 1
            else:
                high = mid - 1

            # Clear cache
            del dummy_input
            torch.cuda.empty_cache()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                high = mid - 1
                torch.cuda.empty_cache()
            else:
                raise

    return optimal_batch_size


def log_metrics(
    logger: logging.Logger,
    metrics: Dict[str, Any],
    step: int,
    prefix: str = "train"
):
    """Log metrics in a structured format"""
    # Console logging
    metric_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                            for k, v in metrics.items()])
    logger.info(f"[Step {step}] {prefix} | {metric_str}")

    # JSON logging for parsing
    json_metrics = {
        'step': step,
        'prefix': prefix,
        'timestamp': time.time(),
        **metrics
    }

    # Write to JSONL file
    log_dir = Path(logger.handlers[0].baseFilename).parent
    jsonl_path = log_dir / f"{prefix}_metrics.jsonl"

    with open(jsonl_path, 'a') as f:
        f.write(json.dumps(json_metrics) + '\n')


def create_optimizer(
    model: torch.nn.Module,
    config: Dict
) -> torch.optim.Optimizer:
    """Create optimizer from configuration"""
    opt_config = config['optimization']

    optimizer_name = opt_config.get('optimizer', 'adamw').lower()
    lr = opt_config.get('lr', 1e-3)
    weight_decay = opt_config.get('weight_decay', 0.01)

    if optimizer_name == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=opt_config.get('betas', (0.9, 0.999)),
            eps=opt_config.get('eps', 1e-8)
        )
    elif optimizer_name == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=opt_config.get('betas', (0.9, 0.999)),
            eps=opt_config.get('eps', 1e-8)
        )
    elif optimizer_name == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=opt_config.get('momentum', 0.9),
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict,
    num_training_steps: Optional[int] = None
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler from configuration"""
    opt_config = config['optimization']
    scheduler_name = opt_config.get('scheduler', 'cosine').lower()

    if scheduler_name == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps or 1000,
            eta_min=opt_config.get('min_lr', 1e-5)
        )
    elif scheduler_name == 'linear':
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=opt_config.get('min_lr', 1e-5) / opt_config.get('lr', 1e-3),
            total_iters=num_training_steps or 1000
        )
    elif scheduler_name == 'none':
        return None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")