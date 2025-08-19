"""Config parsing utilities for OCT foundation model training."""

import os
from pathlib import Path
from typing import Any, Dict
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str) -> DictConfig:
    """Load and parse configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Parsed configuration as OmegaConf DictConfig
    """
    config = OmegaConf.load(config_path)
    
    # Resolve variable interpolations
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
    
    return config


def merge_configs(base_config: DictConfig, override_config: Dict[str, Any]) -> DictConfig:
    """Merge base config with overrides.
    
    Args:
        base_config: Base configuration
        override_config: Dictionary of override values
        
    Returns:
        Merged configuration
    """
    override_conf = OmegaConf.create(override_config)
    merged = OmegaConf.merge(base_config, override_conf)
    return merged


def validate_config(config: DictConfig) -> None:
    """Validate configuration parameters.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = [
        'experiment_name', 'gcs_root', 'manifest_path', 'list_strategy',
        'target_spacing', 'image_size', 'patch_size', 'global_batch_size',
        'per_core_batch_size', 'epochs', 'base_lr'
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate list_strategy
    if config.list_strategy not in ['single_domain', 'multi_domain']:
        raise ValueError(f"Invalid list_strategy: {config.list_strategy}")
    
    # Validate dimensions
    if len(config.target_spacing) != 3:
        raise ValueError("target_spacing must have 3 dimensions [dz, dy, dx]")
    
    if len(config.image_size) != 3:
        raise ValueError("image_size must have 3 dimensions [D, H, W]")
    
    if len(config.patch_size) != 3:
        raise ValueError("patch_size must have 3 dimensions [D, H, W]")
    
    # Validate batch size consistency
    expected_accum = config.global_batch_size // (config.per_core_batch_size * 8)  # 8 TPU cores
    if 'grad_accum_steps' in config and config.grad_accum_steps != expected_accum:
        print(f"Warning: grad_accum_steps ({config.grad_accum_steps}) doesn't match "
              f"expected value ({expected_accum}) for global batch size consistency")


def setup_experiment_dir(config: DictConfig) -> Path:
    """Setup experiment directory structure.
    
    Args:
        config: Configuration containing experiment details
        
    Returns:
        Path to experiment directory
    """
    exp_dir = Path(config.get('exp_dir', f'./experiments/{config.experiment_name}'))
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / 'logs').mkdir(exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    
    return exp_dir