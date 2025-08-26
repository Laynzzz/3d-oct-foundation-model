"""
Training loop module for OCT classification fine-tuning.
Supports linear probe and full fine-tuning modes with metrics tracking.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
from collections import defaultdict
import os

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

from ..models.model import OCTClassificationModel

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'max'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'max' for metrics like accuracy, 'min' for loss
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        self.compare = np.greater if mode == 'max' else np.less
        self.delta = min_delta if mode == 'max' else -min_delta
    
    def __call__(self, score: float) -> bool:
        """Check if early stopping should be triggered."""
        if self.best_score is None:
            self.best_score = score
        elif self.compare(score, self.best_score + self.delta):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            
        self.early_stop = self.counter >= self.patience
        return self.early_stop


class MetricsTracker:
    """Track and compute classification metrics."""
    
    def __init__(self, num_classes: int = 4):
        """Initialize metrics tracker."""
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.all_preds = []
        self.all_labels = []
        self.all_logits = []
        self.total_loss = 0.0
        self.num_samples = 0
    
    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: float):
        """Update metrics with batch results."""
        preds = torch.argmax(logits, dim=1)
        
        self.all_preds.extend(preds.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
        self.all_logits.extend(torch.softmax(logits, dim=1).cpu().numpy())
        self.total_loss += loss * len(labels)
        self.num_samples += len(labels)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        if self.num_samples == 0:
            return {}
        
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_preds)
        y_prob = np.array(self.all_logits)
        
        metrics = {
            'loss': self.total_loss / self.num_samples,
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'macro_f1': f1_score(y_true, y_pred, average='macro'),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted')
        }
        
        # Per-class F1 scores
        f1_per_class = f1_score(y_true, y_pred, average=None)
        for i, f1 in enumerate(f1_per_class):
            metrics[f'f1_class_{i}'] = f1
        
        # AUROC (one-vs-rest)
        try:
            if len(np.unique(y_true)) > 1:  # Need at least 2 classes
                y_true_bin = label_binarize(y_true, classes=list(range(self.num_classes)))
                if y_true_bin.shape[1] == 1:  # Binary case
                    auroc = roc_auc_score(y_true_bin, y_prob[:, 1])
                else:
                    auroc = roc_auc_score(y_true_bin, y_prob, multi_class='ovr', average='macro')
                metrics['auroc_macro'] = auroc
        except Exception as e:
            logger.warning(f"Could not compute AUROC: {e}")
        
        return metrics
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        if self.num_samples == 0:
            return np.zeros((self.num_classes, self.num_classes))
        
        return confusion_matrix(self.all_labels, self.all_preds, labels=list(range(self.num_classes)))


class OCTTrainer:
    """Trainer for OCT classification with linear probe and fine-tuning modes."""
    
    def __init__(
        self,
        model: OCTClassificationModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device = None,
        wandb_run = None,
        use_wandb: bool = False,
        ckpt_dir: Optional[Path] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: OCT classification model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Training device
            wandb_run: W&B run for logging
            use_wandb: Whether to use W&B logging
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device('cpu')
        self.wandb_run = wandb_run
        self.use_wandb = use_wandb
        self.ckpt_dir = ckpt_dir
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_loss_function()
        
        # Setup metrics tracking
        self.num_classes = model.head.num_classes
        self.train_metrics = MetricsTracker(self.num_classes)
        self.val_metrics = MetricsTracker(self.num_classes)
        
        # Setup early stopping
        early_stop_config = config.get('early_stopping', {})
        if early_stop_config.get('enabled', False):
            self.early_stopping = EarlyStopping(
                patience=early_stop_config.get('patience', 10),
                min_delta=early_stop_config.get('min_delta', 0.001),
                mode='max'  # For balanced accuracy
            )
        else:
            self.early_stopping = None
        
        # Training state
        self.epoch = 0
        self.best_val_score = 0.0
        self.history = defaultdict(list)
        
        logger.info(f"Trainer initialized on device: {self.device}")
        logger.info(f"Model parameters: {self.model.count_parameters()}")
    
    def _setup_optimizer(self):
        """Setup optimizer with different learning rates for encoder and head."""
        train_config = self.config.get('train', {})
        
        if self.model.freeze_encoder:
            # Linear probe: only train classification head
            params = self.model.head.parameters()
            lr = train_config.get('lr_head', 1e-3)
            logger.info(f"Linear probe mode: training head only with lr={lr}")
        else:
            # Fine-tuning: different LRs for encoder and head
            lr_encoder = train_config.get('lr_encoder', 1e-5)
            lr_head = train_config.get('lr_head', 5e-4)
            
            params = [
                {'params': self.model.encoder.parameters(), 'lr': lr_encoder},
                {'params': self.model.head.parameters(), 'lr': lr_head}
            ]
            logger.info(f"Fine-tuning mode: encoder lr={lr_encoder}, head lr={lr_head}")
        
        optimizer_name = train_config.get('optimizer', 'AdamW')
        if optimizer_name == 'AdamW':
            self.optimizer = optim.AdamW(
                params,
                weight_decay=train_config.get('weight_decay', 1e-4),
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_name == 'SGD':
            self.optimizer = optim.SGD(
                params,
                momentum=0.9,
                weight_decay=train_config.get('weight_decay', 1e-4),
                nesterov=True
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        train_config = self.config.get('train', {})
        scheduler_type = train_config.get('scheduler', 'cosine')
        
        if scheduler_type == 'cosine':
            total_epochs = train_config.get('epochs', 50)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_epochs
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=train_config.get('step_size', 20),
                gamma=train_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'none':
            self.scheduler = None
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")
    
    def _setup_loss_function(self):
        """Setup loss function with optional class weighting."""
        train_config = self.config.get('train', {})
        class_weights = train_config.get('class_weights', None)
        
        if class_weights == 'auto':
            # This would need to be computed from dataset
            logger.warning("Auto class weights not implemented, using uniform weights")
            self.criterion = nn.CrossEntropyLoss()
        elif isinstance(class_weights, (list, tuple)):
            weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
            logger.info(f"Using class weights: {class_weights}")
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()
        
        epoch_start_time = time.time()
        
        # Check if we're using TPU/XLA
        try:
            from torch_xla.core import xla_model as xm
            using_xla = True
        except ImportError:
            using_xla = False
        
        for batch_idx, (volumes, labels, participant_ids) in enumerate(self.train_loader):
            if len(volumes) == 0:  # Empty batch
                continue
            
            # Per-batch timing (Quick Win #2 from fix plan)
            batch_start_time = time.time()
            io_time = batch_start_time - epoch_start_time if batch_idx == 0 else 0
                
            volumes = volumes.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass timing
            forward_start = time.time()
            self.optimizer.zero_grad()
            logits = self.model(volumes)
            loss = self.criterion(logits, labels)
            forward_time = time.time() - forward_start
            
            # Backward pass timing
            backward_start = time.time()
            loss.backward()
            self.optimizer.step()
            
            # XLA mark step for TPU
            if using_xla:
                xm.mark_step()
            
            backward_time = time.time() - backward_start
            total_batch_time = time.time() - batch_start_time
            
            # Update metrics
            self.train_metrics.update(logits.detach(), labels, loss.item())
            
            # Log batch-level timing to W&B (visible progress)
            if self.use_wandb and WANDB_AVAILABLE:
                global_step = self.epoch * len(self.train_loader) + batch_idx
                
                # Log timing metrics
                timing_logs = {
                    "timing/batch_total_s": total_batch_time,
                    "timing/forward_s": forward_time, 
                    "timing/backward_s": backward_time,
                    "timing/samples_per_sec": len(volumes) / total_batch_time,
                    "batch/loss": loss.item(),
                    "batch/size": len(volumes)
                }
                
                if io_time > 0:
                    timing_logs["timing/io_load_s"] = io_time
                
                # Only log from master process in distributed setting
                is_master = True
                if using_xla:
                    try:
                        import torch_xla.runtime as xr
                        is_master = (xr.global_ordinal() == 0)
                    except:
                        is_master = True
                
                if is_master:
                    wandb.log(timing_logs, step=global_step)
            
            # Log progress every few batches
            if batch_idx % max(1, len(self.train_loader) // 10) == 0:
                logger.info(f"Epoch {self.epoch}, Batch {batch_idx}/{len(self.train_loader)}: "
                           f"loss={loss.item():.4f}, batch_time={total_batch_time:.2f}s, "
                           f"samples/sec={len(volumes)/total_batch_time:.1f}")
            
            epoch_start_time = time.time()  # Reset for next batch I/O timing
            
            # Log batch progress
            if batch_idx % 10 == 0:
                logger.debug(f"Epoch {self.epoch}, Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
        
        # Compute epoch metrics
        train_metrics = self.train_metrics.compute()
        epoch_time = time.time() - epoch_start_time
        
        logger.info(f"Epoch {self.epoch} training completed in {epoch_time:.1f}s")
        logger.info(f"Train metrics: {train_metrics}")
        
        return train_metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate model with TPU-safe master-only approach (Solution C)."""
        self.model.eval()
        
        # Master-only validation to avoid TPU distributed validation issues
        try:
            import torch_xla.core.xla_model as xm
            is_master = xm.is_master_ordinal()
        except:
            # Fallback if not on TPU
            is_master = True
        
        if is_master:
            self.val_metrics.reset()
            
            with torch.no_grad():
                for volumes, labels, participant_ids in self.val_loader:
                    if len(volumes) == 0:  # Empty batch
                        continue
                        
                    volumes = volumes.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Forward pass
                    logits = self.model(volumes)
                    loss = self.criterion(logits, labels)
                    
                    # Update metrics
                    self.val_metrics.update(logits, labels, loss.item())
            
            # Compute validation metrics on master
            metrics = self.val_metrics.compute()
            logger.info(f"Validation metrics: {metrics}")
            
        else:
            # Non-master workers wait and get dummy metrics
            metrics = {
                'loss': 0.0,
                'accuracy': 0.0,
                'balanced_accuracy': 0.0,
                'macro_f1': 0.0,
                'weighted_f1': 0.0,
                'auroc_macro': 0.5
            }
        
        # Synchronize across workers if on TPU
        try:
            import torch_xla.core.xla_model as xm
            if hasattr(xm, 'broadcast_object'):
                metrics = xm.broadcast_object(metrics, src=0)
        except:
            pass  # Not on TPU or broadcast not available
        
        return metrics
    
    def validate_original(self) -> Dict[str, float]:
        """Original validation method - kept as backup."""
        self.model.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            for volumes, labels, participant_ids in self.val_loader:
                if len(volumes) == 0:  # Empty batch
                    continue
                    
                volumes = volumes.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                logits = self.model(volumes)
                loss = self.criterion(logits, labels)
                
                # Update metrics
                self.val_metrics.update(logits, labels, loss.item())
        
        # Compute validation metrics
        val_metrics = self.val_metrics.compute()
        
        logger.info(f"Validation metrics: {val_metrics}")
        return val_metrics
    
    def train(self) -> Dict[str, List[float]]:
        """Full training loop."""
        train_config = self.config.get('train', {})
        total_epochs = train_config.get('epochs', 50)
        unfreeze_at_epoch = self.config.get('model', {}).get('unfreeze_at_epoch', -1)
        
        logger.info(f"Starting training for {total_epochs} epochs")
        logger.info(f"Unfreeze encoder at epoch: {unfreeze_at_epoch if unfreeze_at_epoch > 0 else 'Never'}")
        
        for epoch in range(total_epochs):
            self.epoch = epoch + 1
            
            # Check if we should unfreeze encoder
            if unfreeze_at_epoch > 0 and epoch == unfreeze_at_epoch:
                logger.info(f"Unfreezing encoder at epoch {epoch}")
                self.model.set_freeze_encoder(False)
                # Recreate optimizer with encoder parameters
                self._setup_optimizer()
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Log training metrics immediately to W&B (Solution A: Decouple from validation)
            if self.wandb_run is not None and WANDB_AVAILABLE:
                train_log_dict = {f'train/{k}': v for k, v in train_metrics.items()}
                train_log_dict['epoch'] = self.epoch
                if self.scheduler:
                    train_log_dict['lr/head'] = self.scheduler.get_last_lr()[0]
                self.wandb_run.log(train_log_dict, commit=True)
                logger.info(f"Logged training metrics to W&B for epoch {self.epoch}")
            
            # Track training history
            for key, value in train_metrics.items():
                self.history[f'train_{key}'].append(value)
            
            # Validate with error handling
            try:
                val_metrics = self.validate()
                
                # Track validation history
                for key, value in val_metrics.items():
                    self.history[f'val_{key}'].append(value)
                
                # Log validation metrics to W&B
                if self.wandb_run is not None and WANDB_AVAILABLE:
                    val_log_dict = {f'val/{k}': v for k, v in val_metrics.items()}
                    val_log_dict['epoch'] = self.epoch
                    self.wandb_run.log(val_log_dict, commit=True)
                    logger.info(f"Logged validation metrics to W&B for epoch {self.epoch}")
                    
            except Exception as e:
                logger.error(f"Validation failed at epoch {self.epoch}: {e}")
                # Create dummy validation metrics to prevent crashes
                val_metrics = {
                    'loss': float('nan'),
                    'accuracy': 0.0,
                    'balanced_accuracy': 0.0,
                    'macro_f1': 0.0,
                    'weighted_f1': 0.0,
                    'auroc_macro': 0.5
                }
                
                # Log validation error to W&B
                if self.wandb_run is not None and WANDB_AVAILABLE:
                    error_log_dict = {
                        'val/error': 1,
                        'val/exception': str(e),
                        'epoch': self.epoch
                    }
                    self.wandb_run.log(error_log_dict, commit=True)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Check for best model
            val_score = val_metrics.get('balanced_accuracy', 0.0)
            if val_score > self.best_val_score:
                self.best_val_score = val_score
                logger.info(f"New best validation score: {val_score:.4f}")
                
                # Save best model checkpoint during training
                if hasattr(self, 'ckpt_dir'):
                    best_path = self.ckpt_dir / 'best_checkpoint_during_training.pt'
                    self.save_checkpoint(str(best_path))
            
            # Early stopping check
            if self.early_stopping is not None:
                if self.early_stopping(val_score):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        logger.info(f"Training completed. Best validation score: {self.best_val_score:.4f}")
        return dict(self.history)
    
    def get_predictions(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Get model predictions for a dataset."""
        self.model.eval()
        all_logits = []
        all_labels = []
        all_pids = []
        
        with torch.no_grad():
            for volumes, labels, participant_ids in data_loader:
                if len(volumes) == 0:
                    continue
                    
                volumes = volumes.to(self.device)
                logits = self.model(volumes)
                
                all_logits.extend(torch.softmax(logits, dim=1).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_pids.extend(participant_ids)
        
        logits = np.array(all_logits)
        labels = np.array(all_labels) 
        predictions = np.argmax(logits, axis=1)
        
        return predictions, labels, logits, all_pids
    
    def save_checkpoint(self, checkpoint_path: str, include_optimizer: bool = True):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_score': self.best_val_score,
            'history': dict(self.history),
            'config': self.config
        }
        
        if include_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            if self.scheduler is not None:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")


def create_trainer(
    model: OCTClassificationModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
    use_wandb: bool = False,
    wandb_project: str = None,
    wandb_run_name: str = None,
    ckpt_dir: Optional[Path] = None
) -> OCTTrainer:
    """
    Create trainer instance with optional W&B logging.
    
    Args:
        model: OCT classification model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration
        device: Training device
        use_wandb: Whether to use W&B logging
        wandb_project: W&B project name
        wandb_run_name: W&B run name
        
    Returns:
        Trainer instance
    """
    wandb_run = None
    
    if use_wandb and WANDB_AVAILABLE:
        wandb_run = wandb.init(
            project=wandb_project or config.get('project_name', '3d-oct-classification'),
            name=wandb_run_name,
            config=config
        )
        logger.info(f"W&B logging enabled: {wandb_run.name}")
    elif use_wandb and not WANDB_AVAILABLE:
        logger.warning("W&B requested but not available, skipping logging")
    
    return OCTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        wandb_run=wandb_run,
        use_wandb=use_wandb,
        ckpt_dir=ckpt_dir
    )