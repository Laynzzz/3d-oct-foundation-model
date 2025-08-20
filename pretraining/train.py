#!/usr/bin/env python3
"""
V-JEPA2 3D OCT Foundation Model Pretraining Script

Section 6 Implementation: XLA/TPU Training with distributed training, 
gradient accumulation, mixed precision (BF16), and comprehensive error handling.

Usage:
    python -m torch_xla.distributed.xla_spawn --num_workers=8 pretraining/train.py --config configs/pretrain_vjepa_single_domain.yaml
"""

import argparse
import os
import sys
import time
import math
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.utils.utils as xu
import torch_xla.runtime as xr

import numpy as np
import wandb
import gcsfs
from omegaconf import DictConfig

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config_parser import load_config, validate_config
from utils.logging_setup import setup_logging, setup_wandb, log_system_info, log_model_info, log_training_metrics, MetricsTracker
from data_setup.datasets import OCTDICOMDataset, create_file_lists, collate_fn
from data_setup.transforms import create_pretraining_transforms, create_validation_transforms
from models.vjepa_3d import VJEPA3D


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cosine_lr_scheduler(optimizer, warmup_epochs: int, total_epochs: int, min_lr_ratio: float = 0.01):
    """Get cosine annealing learning rate scheduler with warmup."""
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return epoch / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_optimizer(model: nn.Module, config: DictConfig) -> optim.Optimizer:
    """Create optimizer with weight decay configuration."""
    # Separate parameters with and without weight decay
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # No weight decay for bias, normalization layers, and positional embeddings
            if any(nd in name for nd in ['bias', 'norm', 'pos_embed', 'cls_token']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    
    param_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    
    optimizer = optim.AdamW(param_groups, lr=config.base_lr, weight_decay=config.weight_decay)
    return optimizer


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    step: int,
    loss: float,
    config: DictConfig,
    is_best: bool = False
) -> str:
    """Save model checkpoint to GCS."""
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if local_rank != 0:
        return ""
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'config': dict(config)
    }
    
    # Local save first (faster)
    local_ckpt_dir = Path('/tmp/checkpoints')
    local_ckpt_dir.mkdir(exist_ok=True)
    
    ckpt_name = f"checkpoint_epoch_{epoch:03d}.pt"
    if is_best:
        ckpt_name = f"best_checkpoint.pt"
    
    local_path = local_ckpt_dir / ckpt_name
    torch.save(checkpoint, local_path)
    
    # Upload to GCS
    try:
        fs = gcsfs.GCSFileSystem()
        gcs_path = f"{config.ckpt_dir}/{ckpt_name}"
        fs.mkdirs(config.ckpt_dir, exist_ok=True)
        
        with open(local_path, 'rb') as src, fs.open(gcs_path, 'wb') as dst:
            dst.write(src.read())
        
        logging.getLogger('oct_foundation').info(f"Checkpoint saved to {gcs_path}")
        
        # Save to W&B artifacts if enabled
        if config.wandb.get('log_artifacts', False) and wandb.run is not None:
            artifact_name = config.wandb.get('ckpt_artifact_name', f"{config.experiment_name}-ckpt")
            artifact = wandb.Artifact(
                name=artifact_name,
                type='model',
                description=f"Model checkpoint at epoch {epoch}"
            )
            artifact.add_file(str(local_path))
            wandb.log_artifact(artifact)
        
        return gcs_path
        
    except Exception as e:
        logging.getLogger('oct_foundation').error(f"Failed to save checkpoint to GCS: {e}")
        return str(local_path)


def load_checkpoint(checkpoint_path: str, model: nn.Module, optimizer: optim.Optimizer = None, scheduler=None):
    """Load checkpoint from local or GCS path."""
    logger = logging.getLogger('oct_foundation')
    
    try:
        if checkpoint_path.startswith('gs://'):
            # Download from GCS first
            fs = gcsfs.GCSFileSystem()
            local_path = '/tmp/loaded_checkpoint.pt'
            with fs.open(checkpoint_path, 'rb') as src, open(local_path, 'wb') as dst:
                dst.write(src.read())
            checkpoint_path = local_path
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['step'], checkpoint['loss']
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return 0, 0, float('inf')


def handle_oom_error(config: DictConfig, attempt: int) -> DictConfig:
    """Handle OOM errors by reducing batch size and adjusting gradient accumulation."""
    logger = logging.getLogger('oct_foundation')
    
    if attempt == 1:
        # First attempt: halve per_core_batch_size
        config.per_core_batch_size = max(1, config.per_core_batch_size // 2)
        config.grad_accum_steps = config.global_batch_size // (config.per_core_batch_size * 8)
        logger.warning(f"OOM detected. Reduced per_core_batch_size to {config.per_core_batch_size}, "
                      f"increased grad_accum_steps to {config.grad_accum_steps}")
    
    elif attempt == 2:
        # Second attempt: set batch size to 1, increase grad accumulation
        config.per_core_batch_size = 1
        config.grad_accum_steps = config.global_batch_size // 8
        logger.warning(f"OOM still occurring. Set per_core_batch_size to 1, "
                      f"grad_accum_steps to {config.grad_accum_steps}")
    
    elif attempt == 3:
        # Third attempt: reduce image size
        config.image_size = [64, 320, 320]
        logger.warning(f"Final OOM attempt. Reduced image_size to {config.image_size}")
    
    else:
        logger.error("Unable to resolve OOM after multiple attempts")
        raise RuntimeError("Persistent OOM error")
    
    return config


def create_data_loaders(config: DictConfig) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    logger = logging.getLogger('oct_foundation')
    
    # Create file lists based on strategy
    all_files = create_file_lists(
        manifest_path=config.manifest_path,
        gcs_root=config.gcs_root,
        list_strategy=config.list_strategy
    )
    
    # Split into train/val
    import random
    random.seed(config.seed)
    random.shuffle(all_files)
    val_size = int(0.1 * len(all_files))
    train_files = all_files[val_size:]
    val_files = all_files[:val_size]
    
    logger.info(f"Created file lists: {len(train_files)} training, {len(val_files)} validation")
    
    # Create transforms
    train_transforms = create_pretraining_transforms(
        target_spacing=config.target_spacing,
        image_size=config.image_size,
        mask_ratio=config.mask_ratio
    )
    
    val_transforms = create_validation_transforms(
        target_spacing=config.target_spacing,
        image_size=config.image_size
    )
    
    # Create datasets
    train_dataset = OCTDICOMDataset(
        manifest_path=config.manifest_path,
        gcs_root=config.gcs_root,
        file_list=train_files,
        transforms=train_transforms,
        use_cache=config.get('cache_local', True),
        cache_dir=config.get('cache_dir', '/tmp/oct_cache'),
        target_spacing=config.target_spacing,
        image_size=config.image_size
    )
    
    val_dataset = OCTDICOMDataset(
        manifest_path=config.manifest_path,
        gcs_root=config.gcs_root,
        file_list=val_files,
        transforms=val_transforms,
        use_cache=config.get('cache_local', True),
        cache_dir=config.get('cache_dir', '/tmp/oct_cache'),
        target_spacing=config.target_spacing,
        image_size=config.image_size
    )
    
    # Create distributed samplers
    # Get rank using PyTorch 2.7 compatible method
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=xr.world_size(),
        rank=local_rank,
        shuffle=True,
        drop_last=True
    )
    
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=xr.world_size(),
        rank=local_rank,
        shuffle=False,
        drop_last=False
    )
    
    # Log dataset statistics
    if hasattr(train_dataset, 'get_dataset_stats'):
        train_stats = train_dataset.get_dataset_stats()
        logger.info(f"Training dataset stats: {train_stats}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_core_batch_size,
        sampler=train_sampler,
        num_workers=config.get('workers', 4),
        collate_fn=collate_fn,
        pin_memory=config.get('pin_memory', False),
        prefetch_factor=config.get('prefetch_factor', 2),
        persistent_workers=True if config.get('workers', 4) > 0 else False
    )
    
    # Log dataset statistics
    if hasattr(val_dataset, 'get_dataset_stats'):
        val_stats = val_dataset.get_dataset_stats()
        logger.info(f"Validation dataset stats: {val_stats}")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.per_core_batch_size,
        sampler=val_sampler,
        num_workers=config.get('workers', 4),
        collate_fn=collate_fn,
        pin_memory=config.get('pin_memory', False),
        prefetch_factor=config.get('prefetch_factor', 2),
        persistent_workers=True if config.get('workers', 4) > 0 else False
    )
    
    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    config: DictConfig,
    metrics_tracker: MetricsTracker
) -> float:
    """Train for one epoch."""
    model.train()
    device = xm.xla_device()
    logger = logging.getLogger('oct_foundation')
    
    # Wrap with parallel loader for XLA
    para_loader = pl.ParallelLoader(train_loader, [device])
    
    total_loss = 0.0
    num_batches = 0
    optimizer.zero_grad()
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(para_loader.per_device_loader(device)):
        if batch is None:
            logger.warning(f"Received None batch at step {batch_idx}")
            continue
            
        step_start_time = time.time()
        
        # Check if batch has enough valid samples
        if batch.get('batch_size', 0) < config.per_core_batch_size * 0.5:  # Less than 50% valid
            logger.warning(f"Batch {batch_idx} has only {batch.get('batch_size', 0)}/{config.per_core_batch_size} valid samples")
            # Continue with reduced batch size - this is handled by the collate function
        
        # Forward pass
        try:
            # Mixed precision forward pass
            use_bf16 = getattr(config, 'use_bf16', False) or os.environ.get('XLA_USE_BF16') == '1'
            if use_bf16:
                with torch.autocast(device_type='xla', dtype=torch.bfloat16):
                    # VJEPA3D expects (context_view, target_view, mask)
                    loss, predictions, targets = model(
                        batch['context_view'], 
                        batch['target_view'], 
                        batch['mask']
                    )
            else:
                # VJEPA3D expects (context_view, target_view, mask)
                loss, predictions, targets = model(
                    batch['context_view'], 
                    batch['target_view'], 
                    batch['mask']
                )
            
            # Scale loss for gradient accumulation
            loss = loss / config.grad_accum_steps
            
            # Backward pass
            loss.backward()
            
            total_loss += loss.item() * config.grad_accum_steps
            num_batches += 1
            
            # Gradient accumulation
            if (batch_idx + 1) % config.grad_accum_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # XLA optimizer step
                xm.optimizer_step(optimizer)
                optimizer.zero_grad()
                
                # Update learning rate
                if scheduler is not None:
                    scheduler.step()
            
            # Calculate metrics
            step_time = time.time() - step_start_time
            throughput = config.per_core_batch_size / step_time
            
            metrics_tracker.update(
                loss=loss.item() * config.grad_accum_steps,
                step_time=step_time,
                throughput=throughput
            )
            
            # Logging
            global_step = epoch * len(train_loader) + batch_idx
            
            if (batch_idx + 1) % config.log_every_steps == 0:
                current_lr = scheduler.get_last_lr()[0] if scheduler else config.base_lr
                
                log_training_metrics(
                    step=global_step,
                    epoch=epoch,
                    loss=loss.item() * config.grad_accum_steps,
                    lr=current_lr,
                    throughput=throughput,
                    logger=logger
                )
                
                # Additional W&B metrics
                local_rank = int(os.environ.get('LOCAL_RANK', 0))
                if local_rank == 0 and wandb.run is not None:
                    # Get EMA momentum from target encoder
                    ema_momentum = model.target_encoder.momentum if hasattr(model, 'target_encoder') else 0.0
                    wandb.log({
                        'train/ema_momentum': ema_momentum,
                        'train/grad_norm': torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf')),
                        'train/batch_idx': batch_idx,
                    }, step=global_step)
        
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "resource exhausted" in str(e).lower():
                logger.error(f"OOM error during training: {e}")
                raise e
            else:
                logger.error(f"Runtime error during training: {e}")
                continue
    
    # Final gradient step if needed
    if num_batches % config.grad_accum_steps != 0:
        xm.optimizer_step(optimizer)
        optimizer.zero_grad()
    
    avg_loss = total_loss / max(num_batches, 1)
    epoch_time = time.time() - start_time
    
    logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s, avg_loss: {avg_loss:.6f}")
    
    return avg_loss


def validate_epoch(model: nn.Module, val_loader, epoch: int, config: DictConfig) -> float:
    """Validate for one epoch."""
    model.eval()
    device = xm.xla_device()
    logger = logging.getLogger('oct_foundation')
    
    para_loader = pl.ParallelLoader(val_loader, [device])
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(para_loader.per_device_loader(device)):
            if batch is None:
                continue
            
            try:
                # Forward pass - handle both JEPA and validation formats
                use_bf16 = getattr(config, 'use_bf16', False) or os.environ.get('XLA_USE_BF16') == '1'
                
                if 'context_view' in batch:
                    # JEPA format (from training transforms)
                    if use_bf16:
                        with torch.autocast(device_type='xla', dtype=torch.bfloat16):
                            loss, predictions, targets = model(
                                batch['context_view'], 
                                batch['target_view'], 
                                batch['mask']
                            )
                    else:
                        loss, predictions, targets = model(
                            batch['context_view'], 
                            batch['target_view'], 
                            batch['mask']
                        )
                else:
                    # Validation format - use same image as both context and target
                    image = batch['image']
                    # Create a simple mask (no masking for validation)
                    B, C, D, H, W = image.shape
                    patch_d, patch_h, patch_w = 4, 16, 16  # From config
                    num_patches = (D // patch_d) * (H // patch_h) * (W // patch_w)
                    mask = torch.zeros(B, num_patches, dtype=torch.bool, device=image.device)
                    
                    if use_bf16:
                        with torch.autocast(device_type='xla', dtype=torch.bfloat16):
                            loss, predictions, targets = model(image, image, mask)
                    else:
                        loss, predictions, targets = model(image, image, mask)
                
                total_loss += loss.item()
                num_batches += 1
                
            except RuntimeError as e:
                logger.warning(f"Error during validation: {e}")
                continue
    
    avg_loss = total_loss / max(num_batches, 1)
    
    # Log validation metrics
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if local_rank == 0:
        logger.info(f"Validation epoch {epoch}: avg_loss = {avg_loss:.6f}")
        
        if wandb.run is not None:
            wandb.log({
                'val/loss': avg_loss,
                'val/epoch': epoch
            })
    
    return avg_loss


def main_worker(config: DictConfig):
    """Main training worker function."""
    # Get rank info for PyTorch 2.7 compatibility
    import os
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    is_master = local_rank == 0
    
    # Setup logging
    logger = setup_logging(
        log_level=config.get('log_level', 'INFO'),
        log_file=Path(f"/tmp/train_{local_rank}.log") if not is_master else None
    )
    
    # Set device
    device = xm.xla_device()
    
    # Log system info (master only)
    log_system_info(logger)
    
    # Setup W&B (master only)
    setup_wandb(config)
    
    # Create model
    model = VJEPA3D(
        img_size=config.image_size,
        patch_size=config.patch_size,
        embed_dim=config.get('embed_dim', 768),
        depth=config.get('depth', 12),
        num_heads=config.get('num_heads', 12),
        ema_momentum=config.ema_base
    ).to(device)
    
    # Log model info (master only)
    log_model_info(model, logger)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = get_cosine_lr_scheduler(
        optimizer, 
        warmup_epochs=config.get('warmup_epochs', 10),
        total_epochs=config.epochs
    )
    
    # Load checkpoint if resuming
    start_epoch = 0
    best_loss = float('inf')
    
    if config.get('resume_checkpoint'):
        start_epoch, _, best_loss = load_checkpoint(
            config.resume_checkpoint, model, optimizer, scheduler
        )
        start_epoch += 1  # Start from next epoch
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)
    
    # Metrics tracker
    metrics_tracker = MetricsTracker()
    
    # Training loop
    for epoch in range(start_epoch, config.epochs):
        logger.info(f"Starting epoch {epoch + 1}/{config.epochs}")
        
        # Set epoch for distributed sampler
        train_loader.sampler.set_epoch(epoch)
        
        try:
            # Train
            train_loss = train_epoch(
                model, train_loader, optimizer, scheduler, epoch, config, metrics_tracker
            )
            
            # Validate
            val_loss = validate_epoch(model, val_loader, epoch, config)
            
            # Save checkpoint
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
            
            if (epoch + 1) % config.get('ckpt_every_epochs', 5) == 0 or is_best:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, 
                    epoch * len(train_loader), val_loss, config, is_best
                )
            
            # Reset metrics tracker
            metrics_tracker.reset()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "resource exhausted" in str(e).lower():
                logger.error(f"OOM error in epoch {epoch}: {e}")
                # In distributed setting, need to handle OOM collectively
                raise e
            else:
                logger.error(f"Runtime error in epoch {epoch}: {e}")
                continue
    
    # Final checkpoint
    save_checkpoint(
        model, optimizer, scheduler, config.epochs - 1,
        config.epochs * len(train_loader), best_loss, config, False
    )
    
    logger.info("Training completed successfully!")


def train_with_error_handling(config: DictConfig, max_oom_attempts: int = 3):
    """Train with automatic OOM error handling."""
    logger = logging.getLogger('oct_foundation')
    
    for attempt in range(max_oom_attempts):
        try:
            main_worker(config)
            return  # Success
            
        except RuntimeError as e:
            if attempt < max_oom_attempts - 1 and ("out of memory" in str(e).lower() or "resource exhausted" in str(e).lower()):
                logger.warning(f"OOM attempt {attempt + 1}, retrying with reduced settings...")
                config = handle_oom_error(config, attempt + 1)
                
                # Clear cache and reset XLA
                xm.mark_step()
                torch._C._xla._XLAC._xla_sync_multi([str(xm.xla_device())], [], [])
                
            else:
                raise e




def _mp_fn(index):
    """Main worker function for XLA multiprocessing"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='V-JEPA2 3D OCT Foundation Model Pretraining')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load and validate config
    config = load_config(args.config)
    validate_config(config)
    
    # Set random seed
    set_seed(config.get('seed', 42))
    
    # Start training with error handling
    train_with_error_handling(config)


if __name__ == '__main__':
    # Fix multiprocessing compatibility issues with Python 3.11 + XLA 2.7.0
    import multiprocessing
    multiprocessing.set_start_method('forkserver', force=True)
    
    # Use XLA multiprocessing for distributed training
    xmp.spawn(_mp_fn, nprocs=None)