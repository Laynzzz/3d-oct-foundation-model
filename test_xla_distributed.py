#!/usr/bin/env python3
"""
Test XLA distributed training with proper initialization
"""
import os
import sys
from pathlib import Path
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.vjepa_3d import VJEPA3D

def _mp_fn(index):
    """Main worker function for XLA multiprocessing"""
    print(f"Worker {index}: Starting")
    
    # Get device
    device = xm.xla_device()
    print(f"Worker {index}: Using device {device}")
    
    # Create small test model
    model = VJEPA3D(
        img_size=[64, 384, 384],
        patch_size=[4, 16, 16],
        embed_dim=384,
        depth=6,
        num_heads=6,
        ema_momentum=0.996
    ).to(device)
    
    print(f"Worker {index}: Model created, params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create test input
    batch_size = 1
    test_input = torch.randn(batch_size, 1, 64, 384, 384).to(device)
    test_mask = torch.ones(batch_size, 64//4 * 384//16 * 384//16).bool().to(device)
    test_mask[:, :int(0.6 * test_mask.size(1))] = False  # 60% masked
    
    print(f"Worker {index}: Starting forward pass")
    
    # Forward pass
    with torch.no_grad():
        loss, preds, targets = model(test_input, test_mask)
    
    print(f"Worker {index}: Forward pass complete, loss: {loss.item():.6f}")
    
    # Synchronize across all replicas
    xm.rendezvous('test_complete')
    
    if index == 0:
        print("All workers completed successfully!")

if __name__ == "__main__":
    # Run with XLA multiprocessing (nprocs=None uses all available devices)
    xmp.spawn(_mp_fn, nprocs=None)