"""
Classification head module for fine-tuning.
Supports linear probe and MLP head configurations.
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ClassificationHead(nn.Module):
    """Classification head for OCT diabetes status prediction."""
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 4,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_batch_norm: bool = False,
        activation: str = 'relu'
    ):
        """
        Initialize classification head.
        
        Args:
            embed_dim: Input embedding dimension from encoder
            num_classes: Number of output classes (4 for diabetes classification)
            hidden_dim: Hidden dimension for MLP head (None for linear probe)
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'gelu', 'leaky_relu')
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Build head architecture
        if hidden_dim is None or hidden_dim == 0:
            # Linear probe: single linear layer
            self.head = nn.Sequential(
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(embed_dim, num_classes)
            )
            logger.info(f"Created linear probe head: {embed_dim} -> {num_classes}")
            
        else:
            # MLP head with hidden layer
            activation_layer = self._get_activation(activation)
            
            layers = [nn.Linear(embed_dim, hidden_dim)]
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.extend([
                activation_layer,
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, num_classes)
            ])
            
            self.head = nn.Sequential(*[layer for layer in layers if layer is not None])
            logger.info(f"Created MLP head: {embed_dim} -> {hidden_dim} -> {num_classes}")
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        
        if activation.lower() not in activations:
            logger.warning(f"Unknown activation '{activation}', using ReLU")
            return nn.ReLU()
        
        return activations[activation.lower()]
    
    def _init_weights(self):
        """Initialize layer weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier normal initialization for linear layers
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.
        
        Args:
            x: Input features [B, embed_dim]
            
        Returns:
            Logits [B, num_classes]
        """
        if x.dim() != 2 or x.size(1) != self.embed_dim:
            raise ValueError(f"Expected input shape [B, {self.embed_dim}], got {x.shape}")
        
        return self.head(x)


class AdaptiveClassificationHead(nn.Module):
    """Adaptive classification head that can switch between linear and MLP modes."""
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 4,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize adaptive classification head.
        
        Args:
            embed_dim: Input embedding dimension
            num_classes: Number of output classes
            hidden_dim: Hidden dimension for MLP mode
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Linear probe components
        self.linear_dropout = nn.Dropout(dropout)
        self.linear_classifier = nn.Linear(embed_dim, num_classes)
        
        # MLP components
        self.mlp_layer1 = nn.Linear(embed_dim, hidden_dim)
        self.mlp_bn = nn.BatchNorm1d(hidden_dim)
        self.mlp_activation = nn.GELU()
        self.mlp_dropout = nn.Dropout(dropout)
        self.mlp_classifier = nn.Linear(hidden_dim, num_classes)
        
        self.mode = 'linear'  # Default to linear probe
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for all layers."""
        for module in [self.linear_classifier, self.mlp_layer1, self.mlp_classifier]:
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        
        nn.init.constant_(self.mlp_bn.weight, 1)
        nn.init.constant_(self.mlp_bn.bias, 0)
    
    def set_mode(self, mode: str):
        """Switch between linear probe and MLP modes."""
        if mode not in ['linear', 'mlp']:
            raise ValueError(f"Mode must be 'linear' or 'mlp', got {mode}")
        
        self.mode = mode
        logger.info(f"Classification head mode set to: {mode}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with mode switching."""
        if self.mode == 'linear':
            # Linear probe path
            x = self.linear_dropout(x)
            return self.linear_classifier(x)
        else:
            # MLP path
            x = self.mlp_layer1(x)
            x = self.mlp_bn(x)
            x = self.mlp_activation(x)
            x = self.mlp_dropout(x)
            return self.mlp_classifier(x)


class WeightedClassificationHead(ClassificationHead):
    """Classification head with built-in class weighting for imbalanced datasets."""
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = 4,
        class_weights: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Initialize weighted classification head.
        
        Args:
            embed_dim: Input embedding dimension
            num_classes: Number of output classes
            class_weights: Tensor of class weights [num_classes]
            **kwargs: Additional arguments for parent class
        """
        super().__init__(embed_dim=embed_dim, num_classes=num_classes, **kwargs)
        
        if class_weights is not None:
            if len(class_weights) != num_classes:
                raise ValueError(f"Expected {num_classes} class weights, got {len(class_weights)}")
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None
    
    def get_loss_fn(self) -> nn.Module:
        """Get cross-entropy loss with class weights."""
        if self.class_weights is not None:
            return nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            return nn.CrossEntropyLoss()


def create_classification_head(config: dict) -> nn.Module:
    """
    Create classification head from configuration.
    
    Args:
        config: Configuration dictionary with head parameters
        
    Returns:
        Classification head module
    """
    head_config = config.get('head', {})
    
    return ClassificationHead(
        embed_dim=config.get('emb_dim', 768),
        num_classes=len(config.get('classes', {}).get('mapping', {})) or 4,
        hidden_dim=head_config.get('hidden', None),
        dropout=head_config.get('dropout', 0.1),
        use_batch_norm=head_config.get('batch_norm', False),
        activation=head_config.get('activation', 'relu')
    )


def create_weighted_head(config: dict, class_weights: torch.Tensor) -> WeightedClassificationHead:
    """
    Create weighted classification head from configuration.
    
    Args:
        config: Configuration dictionary
        class_weights: Class weights tensor
        
    Returns:
        Weighted classification head
    """
    head_config = config.get('head', {})
    
    return WeightedClassificationHead(
        embed_dim=config.get('emb_dim', 768),
        num_classes=len(config.get('classes', {}).get('mapping', {})) or 4,
        class_weights=class_weights,
        hidden_dim=head_config.get('hidden', None),
        dropout=head_config.get('dropout', 0.1),
        use_batch_norm=head_config.get('batch_norm', False),
        activation=head_config.get('activation', 'relu')
    )


def count_parameters(head: nn.Module) -> tuple:
    """
    Count parameters in classification head.
    
    Args:
        head: Classification head module
        
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in head.parameters())
    trainable = sum(p.numel() for p in head.parameters() if p.requires_grad)
    return total, trainable