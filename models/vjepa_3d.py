import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PatchEmbed3D(nn.Module):
    """3D patch embedding for OCT volumes."""
    
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (64, 384, 384),
        patch_size: Tuple[int, int, int] = (4, 16, 16),
        in_chans: int = 1,
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1], 
            img_size[2] // patch_size[2]
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        
        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        assert D == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2], \
            f"Input image size ({D}x{H}x{W}) doesn't match model ({self.img_size[0]}x{self.img_size[1]}x{self.img_size[2]})"
        
        x = self.proj(x)  # [B, embed_dim, D', H', W']
        x = rearrange(x, 'b e d h w -> b (d h w) e')  # [B, num_patches, embed_dim]
        return x


class Attention3D(nn.Module):
    """Multi-head self-attention for 3D ViT."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block3D(nn.Module):
    """Transformer block for 3D ViT."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        drop: float = 0.,
        attn_drop: float = 0.,
        drop_path: float = 0.,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention3D(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_hidden_dim,
            act_layer=act_layer, drop=drop
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class VisionTransformer3D(nn.Module):
    """3D Vision Transformer backbone for V-JEPA2."""
    
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (64, 384, 384),
        patch_size: Tuple[int, int, int] = (4, 16, 16),
        in_chans: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbed3D(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block3D(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights_module)
        
    def _init_weights_module(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x


class Predictor(nn.Module):
    """2-layer MLP predictor for V-JEPA2."""
    
    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dim: Optional[int] = None,
        norm_layer: nn.Module = nn.BatchNorm1d
    ):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            norm_layer(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, embed_dim]
        B, N, D = x.shape
        x = x.view(B * N, D)  # Flatten for BatchNorm1d
        x = self.net(x)
        x = x.view(B, N, D)  # Reshape back
        return x


class EMAEncoder(nn.Module):
    """Target encoder with exponential moving average (EMA) momentum."""
    
    def __init__(self, encoder: nn.Module, momentum: float = 0.996):
        super().__init__()
        self.encoder = encoder
        self.momentum = momentum
        
        # Copy parameters and disable gradients
        for param in self.encoder.parameters():
            param.requires_grad = False
            
    @torch.no_grad()
    def update(self, context_encoder: nn.Module, momentum: Optional[float] = None):
        """Update target encoder with EMA momentum."""
        m = momentum if momentum is not None else self.momentum
        
        for param_target, param_context in zip(
            self.encoder.parameters(), context_encoder.parameters()
        ):
            param_target.data.mul_(m).add_(param_context.data, alpha=1.0 - m)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def cosine_ema_schedule(step: int, total_steps: int, base_momentum: float = 0.996) -> float:
    """Cosine EMA momentum schedule from base to 1.0."""
    return 1 - (1 - base_momentum) * (math.cos(math.pi * step / total_steps) + 1) / 2


class NormalizedMSELoss(nn.Module):
    """Normalized MSE loss for V-JEPA2 (cosine-style regression)."""
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: [B, N, D] predicted features
            targets: [B, N, D] target features
            mask: [B, N] binary mask (1 for masked patches to predict)
        """
        # L2 normalize features
        pred_norm = F.normalize(predictions, p=2, dim=-1, eps=self.eps)
        target_norm = F.normalize(targets, p=2, dim=-1, eps=self.eps)
        
        # Compute MSE only on masked patches
        loss = F.mse_loss(pred_norm, target_norm, reduction='none')  # [B, N, D]
        loss = loss.mean(dim=-1)  # [B, N]
        
        # Apply mask and average
        masked_loss = loss * mask
        return masked_loss.sum() / (mask.sum() + self.eps)


class VJEPA3D(nn.Module):
    """Complete V-JEPA2 3D model for retinal OCT pretraining."""
    
    def __init__(
        self,
        img_size: Tuple[int, int, int] = (64, 384, 384),
        patch_size: Tuple[int, int, int] = (4, 16, 16),
        in_chans: int = 1,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        ema_momentum: float = 0.996,
        predictor_hidden_dim: Optional[int] = None
    ):
        super().__init__()
        
        # Context encoder (trainable)
        self.context_encoder = VisionTransformer3D(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate
        )
        
        # Target encoder (EMA copy)
        target_encoder = VisionTransformer3D(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=0.,  # No dropout for target
            attn_drop_rate=0., drop_path_rate=0.
        )
        self.target_encoder = EMAEncoder(target_encoder, momentum=ema_momentum)
        
        # Initialize target encoder with context encoder weights
        self.target_encoder.update(self.context_encoder, momentum=0.0)
        
        # Predictor network
        self.predictor = Predictor(
            embed_dim=embed_dim,
            hidden_dim=predictor_hidden_dim or embed_dim
        )
        
        # Loss function
        self.criterion = NormalizedMSELoss()
        
        # Store config
        self.patch_grid_size = self.context_encoder.patch_embed.grid_size
        self.num_patches = self.context_encoder.patch_embed.num_patches
        
    def update_target_encoder(self, momentum: Optional[float] = None):
        """Update target encoder with EMA momentum."""
        self.target_encoder.update(self.context_encoder, momentum)
        
    def forward(
        self, 
        context_view: torch.Tensor, 
        target_view: torch.Tensor, 
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for V-JEPA2 training.
        
        Args:
            context_view: [B, C, D, H, W] context OCT volume
            target_view: [B, C, D, H, W] target OCT volume  
            mask: [B, num_patches] binary mask (1 for patches to predict)
            
        Returns:
            loss: scalar loss value
            predictions: [B, num_patches, embed_dim] predicted features
            targets: [B, num_patches, embed_dim] target features
        """
        with torch.no_grad():
            # Target encoder (no gradients)
            target_features = self.target_encoder(target_view)  # [B, N, D]
            
        # Context encoder (with gradients)
        context_features = self.context_encoder(context_view)  # [B, N, D]
        
        # Predictor maps context to target space
        predictions = self.predictor(context_features)  # [B, N, D]
        
        # Compute loss on masked patches only
        loss = self.criterion(predictions, target_features, mask)
        
        return loss, predictions, target_features
        
    def encode_context(self, x: torch.Tensor) -> torch.Tensor:
        """Encode context view (for inference/feature extraction)."""
        return self.context_encoder(x)
        
    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        """Encode target view (for inference/feature extraction)."""
        return self.target_encoder(x)