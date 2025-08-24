"""
Training runner script for OCT classification fine-tuning.
Supports Hydra configuration management and multi-run sweeps.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np
import random

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    print("Warning: Hydra not available. Install with: pip install hydra-core")

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from finetuning.data.labels import process_labels
from finetuning.data.dataset import create_dataloader, create_debug_dataloader
from finetuning.data.transforms import VJepa2Transforms
from finetuning.models.model import create_model_from_checkpoint
from finetuning.train.loop import create_trainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # For deterministic training (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


def setup_device(config: Dict[str, Any]) -> torch.device:
    """Setup training device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    
    return device


def create_data_loaders(config: Dict[str, Any]) -> tuple:
    """Create train and validation data loaders."""
    data_config = config['data']
    paths_config = config['paths']
    
    # Process labels
    train_df, val_df, test_df, class_to_idx, class_weights = process_labels(
        paths_config['labels_tsv']
    )
    
    logger.info(f"Dataset sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Create transforms
    train_transforms = VJepa2Transforms(
        target_shape=tuple(data_config['augment']['resize']),
        augment=data_config['augment'].get('flip', False) or data_config['augment'].get('intensity_jitter', False)
    )
    
    val_transforms = VJepa2Transforms(
        target_shape=tuple(data_config['augment']['resize']),
        augment=False
    )
    
    # Check if debug mode
    debug_mode = config.get('debug_mode', False)
    
    if debug_mode:
        logger.info("Using debug mode with dummy data")
        train_loader = create_debug_dataloader(
            train_df.head(8),  # Use small subset
            batch_size=data_config['batch_size'],
            transforms=train_transforms,
            shuffle=True
        )
        
        val_loader = create_debug_dataloader(
            val_df.head(4),
            batch_size=data_config['val_batch_size'],
            transforms=val_transforms,
            shuffle=False
        )
    else:
        # Real data loaders
        train_loader = create_dataloader(
            train_df,
            batch_size=data_config['batch_size'],
            transforms=train_transforms,
            shuffle=True,
            num_workers=data_config['num_workers'],
            cache_dir=data_config.get('cache_dir')
        )
        
        val_loader = create_dataloader(
            val_df,
            batch_size=data_config['val_batch_size'],
            transforms=val_transforms,
            shuffle=False,
            num_workers=data_config['num_workers'],
            cache_dir=data_config.get('cache_dir')
        )
    
    return train_loader, val_loader, class_to_idx, class_weights


def train_model(config: Dict[str, Any]) -> Dict[str, Any]:
    """Main training function."""
    # Set random seed
    set_seed(config.get('seed', 42))
    
    # Setup device
    device = setup_device(config)
    
    # Create data loaders
    train_loader, val_loader, class_to_idx, class_weights = create_data_loaders(config)
    
    # Update config with discovered class information
    if 'classes' not in config:
        config['classes'] = {}
    config['classes']['mapping'] = class_to_idx
    
    # Create model
    checkpoint_path = config['paths']['checkpoint_path']
    freeze_encoder = config['model']['freeze_encoder']
    
    logger.info(f"Creating model from checkpoint: {os.path.basename(checkpoint_path)}")
    logger.info(f"Freeze encoder: {freeze_encoder}")
    
    model = create_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        config=config,
        freeze_encoder=freeze_encoder,
        device=device
    )
    
    # Setup W&B logging
    wandb_run = None
    log_config = config.get('log', {})
    
    if log_config.get('wandb', False) and WANDB_AVAILABLE:
        # Generate run name
        checkpoint_name = os.path.basename(checkpoint_path).replace('.pt', '').replace('best_checkpoint_', '')
        mode = "linear_probe" if freeze_encoder else "finetune"
        run_name = f"{checkpoint_name}_{mode}"
        
        wandb_run = wandb.init(
            project=log_config.get('wandb_project', '3d-oct-foundation-model'),
            entity=log_config.get('wandb_entity'),
            name=run_name,
            config=dict(config)
        )
        logger.info(f"W&B logging enabled: {wandb_run.name}")
    
    # Create trainer
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        use_wandb=False,  # Already initialized above
        wandb_run=wandb_run
    )
    
    # Train model
    logger.info("Starting training...")
    history = trainer.train()
    
    # Save results
    ckpt_dir = Path(log_config.get('ckpt_dir', './runs/default'))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best model if requested
    if log_config.get('save_best', True):
        best_model_path = ckpt_dir / 'best_model.pt'
        trainer.save_checkpoint(str(best_model_path), include_optimizer=False)
    
    # Save training history
    history_path = ckpt_dir / 'training_history.pt'
    torch.save(history, history_path)
    logger.info(f"Training history saved to {history_path}")
    
    # Get final predictions on validation set
    predictions, labels, logits, participant_ids = trainer.get_predictions(val_loader)
    
    # Save predictions
    predictions_data = {
        'predictions': predictions,
        'labels': labels,
        'logits': logits,
        'participant_ids': participant_ids,
        'class_to_idx': class_to_idx
    }
    
    pred_path = ckpt_dir / 'val_predictions.pt'
    torch.save(predictions_data, pred_path)
    logger.info(f"Validation predictions saved to {pred_path}")
    
    # Final metrics
    final_train_metrics = {k: v[-1] for k, v in history.items() if k.startswith('train_')}
    final_val_metrics = {k: v[-1] for k, v in history.items() if k.startswith('val_')}
    
    logger.info("Training completed!")
    logger.info(f"Final train metrics: {final_train_metrics}")
    logger.info(f"Final val metrics: {final_val_metrics}")
    logger.info(f"Best validation balanced accuracy: {trainer.best_val_score:.4f}")
    
    # Close W&B run
    if wandb_run is not None:
        wandb.finish()
    
    return {
        'history': history,
        'best_val_score': trainer.best_val_score,
        'final_train_metrics': final_train_metrics,
        'final_val_metrics': final_val_metrics,
        'checkpoint_dir': str(ckpt_dir)
    }


@hydra.main(version_base=None, config_path="../../configs", config_name="cls_linear_probe")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration management."""
    if not HYDRA_AVAILABLE:
        raise ImportError("Hydra is required for configuration management. Install with: pip install hydra-core")
    
    # Convert OmegaConf to dict for easier handling
    config = OmegaConf.to_container(cfg, resolve=True)
    
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Check if checkpoint exists
    checkpoint_path = config['paths']['checkpoint_path']
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    try:
        results = train_model(config)
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def main_simple(config_path: str, checkpoint_path: Optional[str] = None):
    """Simple entry point without Hydra for programmatic use."""
    import yaml
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override checkpoint path if provided
    if checkpoint_path is not None:
        config['paths']['checkpoint_path'] = checkpoint_path
    
    logger.info(f"Loaded config from {config_path}")
    
    # Check checkpoint exists
    checkpoint_path = config['paths']['checkpoint_path']
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        results = train_model(config)
        logger.info("Training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False


if __name__ == "__main__":
    if HYDRA_AVAILABLE:
        main()
    else:
        # Fallback mode without Hydra
        import argparse
        
        parser = argparse.ArgumentParser(description="Train OCT classification model")
        parser.add_argument("--config", default="configs/cls_linear_probe.yaml", help="Config file path")
        parser.add_argument("--checkpoint", help="Override checkpoint path")
        
        args = parser.parse_args()
        
        success = main_simple(args.config, args.checkpoint)
        sys.exit(0 if success else 1)