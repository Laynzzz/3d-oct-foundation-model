"""Logging setup utilities for OCT foundation model training."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional
import torch
import wandb
import torch_xla.core.xla_model as xm
from omegaconf import DictConfig


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        log_format: Optional custom log format
        
    Returns:
        Configured logger
    """
    if log_format is None:
        log_format = '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[]
    )
    
    logger = logging.getLogger('oct_foundation')
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
    
    return logger


def setup_wandb(config: DictConfig, resume: bool = False) -> None:
    """Setup Weights & Biases logging.
    
    Args:
        config: Configuration containing wandb settings
        resume: Whether to resume existing run
    """
    if not xm.is_master_ordinal():
        return
    
    wandb_config = config.get('wandb', {})
    
    # Extract run name from experiment config
    run_name = f"{config.experiment_name}"
    if 'git_sha' in config:
        run_name += f"-{config.git_sha[:7]}"
    
    wandb.init(
        project=wandb_config.get('project', 'oct-foundation'),
        entity=wandb_config.get('entity', None),
        name=run_name,
        config=dict(config),
        resume=resume,
        settings=wandb.Settings(start_method="fork")
    )


def log_system_info(logger: logging.Logger) -> None:
    """Log system and environment information.
    
    Args:
        logger: Logger instance
    """
    if not xm.is_master_ordinal():
        return
        
    import torch
    import torch_xla
    
    logger.info("=== System Information ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"XLA version: {torch_xla.__version__}")
    logger.info(f"XLA device: {xm.xla_device()}")
    logger.info(f"Number of XLA devices: {xm.xrt_world_size()}")
    logger.info(f"Process ordinal: {xm.get_ordinal()}")
    logger.info(f"Local ordinal: {xm.get_local_ordinal()}")
    
    # Environment variables
    xla_env_vars = {k: v for k, v in os.environ.items() if 'XLA' in k or 'TPU' in k}
    if xla_env_vars:
        logger.info("XLA/TPU Environment Variables:")
        for key, value in xla_env_vars.items():
            logger.info(f"  {key}: {value}")


def log_model_info(model: torch.nn.Module, logger: logging.Logger) -> None:
    """Log model architecture information.
    
    Args:
        model: PyTorch model
        logger: Logger instance
    """
    if not xm.is_master_ordinal():
        return
        
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("=== Model Information ===")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Model size (MB): {total_params * 4 / 1024 / 1024:.2f}")


def log_training_metrics(
    step: int,
    epoch: int,
    loss: float,
    lr: float,
    throughput: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
    log_wandb: bool = True
) -> None:
    """Log training metrics.
    
    Args:
        step: Training step
        epoch: Training epoch
        loss: Loss value
        lr: Learning rate
        throughput: Optional throughput (samples/sec/core)
        logger: Optional logger instance
        log_wandb: Whether to log to wandb
    """
    if not xm.is_master_ordinal():
        return
    
    metrics = {
        'train/loss': loss,
        'train/lr': lr,
        'train/epoch': epoch,
        'train/step': step
    }
    
    if throughput is not None:
        metrics['train/throughput_samples_per_sec_per_core'] = throughput
    
    # Log to console
    if logger is not None:
        msg = f"Step {step} | Epoch {epoch} | Loss: {loss:.6f} | LR: {lr:.2e}"
        if throughput is not None:
            msg += f" | Throughput: {throughput:.2f} samples/s/core"
        logger.info(msg)
    
    # Log to wandb
    if log_wandb and wandb.run is not None:
        wandb.log(metrics, step=step)


class MetricsTracker:
    """Class to track and aggregate training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all tracked metrics."""
        self.losses = []
        self.step_times = []
        self.throughputs = []
    
    def update(self, loss: float, step_time: float, throughput: float):
        """Update metrics with new values."""
        self.losses.append(loss)
        self.step_times.append(step_time)
        self.throughputs.append(throughput)
    
    def get_averages(self):
        """Get average values of tracked metrics."""
        if not self.losses:
            return None
            
        return {
            'avg_loss': sum(self.losses) / len(self.losses),
            'avg_step_time': sum(self.step_times) / len(self.step_times),
            'avg_throughput': sum(self.throughputs) / len(self.throughputs)
        }