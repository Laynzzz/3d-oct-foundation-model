#!/usr/bin/env python3
"""Simple TPU test to debug initialization issues with PyTorch 2.7 + torchrun"""

import os
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

def test_initialization():
    """Test XLA initialization sequence"""
    print("🔧 Testing XLA initialization with PyTorch 2.7 + torchrun")
    print("=" * 60)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"XLA version: {torch_xla.__version__}")
    
    # Check environment
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    print(f"Environment: LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}")
    
    try:
        # Test device access
        device = xm.xla_device()
        print(f"✅ XLA device: {device}")
        
        # Test device count
        local_devices = xr.local_device_count()
        print(f"✅ Local device count: {local_devices}")
        
        # Test ordinal APIs
        ordinal = xm.get_ordinal()
        local_ordinal = xm.get_local_ordinal()
        is_master = xm.is_master_ordinal()
        print(f"✅ Ordinals: global={ordinal}, local={local_ordinal}, is_master={is_master}")
        
        # Test simple tensor operation
        x = torch.randn(4, 4, device=device)
        y = torch.matmul(x, x)
        xm.mark_step()
        print(f"✅ Simple tensor operation successful: {y.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_initialization()
    exit(0 if success else 1)