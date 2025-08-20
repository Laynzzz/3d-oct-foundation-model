#!/usr/bin/env python3
"""
TPU PyTorch 2.7.1 / XLA 2.7.0 Compatibility Test
Tests specific features and APIs for the exact versions being used
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_pytorch_xla_versions():
    """Test PyTorch and XLA versions"""
    print("🔧 Testing PyTorch 2.7.1 / XLA 2.7.0 Compatibility")
    print("=" * 60)
    
    try:
        import torch
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        import torch_xla.distributed.parallel_loader as pl
        
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ XLA version: {torch_xla.__version__}")
        
        # Check if versions match expected
        if torch.__version__.startswith('2.7'):
            print("✅ PyTorch 2.7.x confirmed")
        else:
            print(f"⚠️  Expected PyTorch 2.7.x, got {torch.__version__}")
            
        if torch_xla.__version__.startswith('2.7'):
            print("✅ XLA 2.7.x confirmed")
        else:
            print(f"⚠️  Expected XLA 2.7.x, got {torch_xla.__version__}")
        
        # Test XLA runtime initialization
        print(f"✅ XLA runtime initialized: {torch_xla._XLAC.is_runtime_initialized()}")
        
        # Test device count API (new in 2.7)
        device_count = xr.device_count()
        print(f"✅ TPU device count: {device_count}")
        
        if device_count == 0:
            print("❌ No TPU devices found!")
            return False
        
        # Test world size API (new in 2.7)
        world_size = xr.world_size()
        print(f"✅ World size: {world_size}")
        
        # Test device API
        device = xm.xla_device()
        print(f"✅ XLA device: {device}")
        
        # Test ordinal APIs
        ordinal = xm.get_ordinal()
        is_master = xm.is_master_ordinal()
        print(f"✅ Process ordinal: {ordinal}, is_master: {is_master}")
        
        return True
    except Exception as e:
        print(f"❌ Version test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_on_tpu():
    """Test V-JEPA model on TPU with PyTorch 2.7"""
    print("\n🧠 Testing V-JEPA Model on TPU (PyTorch 2.7)")
    print("=" * 60)
    
    try:
        import torch
        import torch_xla.core.xla_model as xm
        from models.vjepa_3d import VJEPA3D
        
        device = xm.xla_device()
        print(f"Using device: {device}")
        
        # Create model optimized for PyTorch 2.7
        model = VJEPA3D(
            img_size=(16, 64, 64),
            patch_size=(4, 16, 16),
            embed_dim=256,
            depth=4,
            num_heads=8,
            ema_momentum=0.996
        ).to(device)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created: {param_count:,} parameters")
        
        # Test forward pass with proper shapes
        batch_size = 2
        context = torch.randn(batch_size, 1, 16, 64, 64, device=device)
        target = torch.randn(batch_size, 1, 16, 64, 64, device=device)
        
        # Create mask - should be [B, num_patches]
        num_patches = (16//4) * (64//16) * (64//16)  # 64 patches
        mask = torch.randint(0, 2, (batch_size, num_patches), device=device).bool()
        
        print(f"   Input shapes: context={context.shape}, target={target.shape}, mask={mask.shape}")
        
        # Forward pass
        start_time = time.time()
        with torch.no_grad():
            loss, predictions, targets = model(context, target, mask)
        
        # Force synchronization (important for TPU timing)
        xm.mark_step()
        elapsed = time.time() - start_time
        
        print(f"✅ Forward pass successful!")
        print(f"   Loss: {loss.item():.6f}")
        print(f"   Predictions: {predictions.shape}")
        print(f"   Targets: {targets.shape}")
        print(f"   Time: {elapsed:.3f}s")
        
        # Test gradient computation
        model.train()
        loss, _, _ = model(context, target, mask)
        loss.backward()
        
        # Check gradients
        grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        print(f"✅ Backward pass successful, grad norm: {grad_norm:.6f}")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading_apis():
    """Test data loading with PyTorch 2.7 APIs"""
    print("\n📊 Testing Data Loading APIs")
    print("=" * 60)
    
    try:
        from torch.utils.data import DataLoader, DistributedSampler
        import torch_xla.runtime as xr
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
        
        # Test distributed sampler with new APIs
        dummy_dataset = list(range(100))
        
        sampler = DistributedSampler(
            dummy_dataset,
            num_replicas=xr.world_size(),
            rank=xm.get_ordinal(),
            shuffle=True
        )
        
        dataloader = DataLoader(
            dummy_dataset,
            batch_size=4,
            sampler=sampler,
            num_workers=1,
            drop_last=True
        )
        
        print(f"✅ DistributedSampler created with world_size={xr.world_size()}")
        
        # Test ParallelLoader (XLA specific)
        device = xm.xla_device()
        para_loader = pl.ParallelLoader(dataloader, [device])
        
        print("✅ ParallelLoader created")
        
        # Test iteration
        for i, batch in enumerate(para_loader):
            if i >= 2:  # Just test first few batches
                break
            print(f"   Batch {i}: {len(batch)} items")
        
        print("✅ Data loading iteration successful")
        
        return True
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optimizer_apis():
    """Test optimizer step API for PyTorch 2.7"""
    print("\n⚙️  Testing Optimizer APIs")
    print("=" * 60)
    
    try:
        import torch
        import torch.optim as optim
        import torch_xla.core.xla_model as xm
        
        device = xm.xla_device()
        
        # Create simple model
        model = torch.nn.Linear(10, 1).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        
        # Test data
        x = torch.randn(4, 10, device=device)
        y = torch.randn(4, 1, device=device)
        
        # Forward and backward
        pred = model(x)
        loss = torch.nn.functional.mse_loss(pred, y)
        loss.backward()
        
        # Test XLA optimizer step (critical for TPU)
        xm.optimizer_step(optimizer)
        optimizer.zero_grad()
        
        print(f"✅ XLA optimizer step successful")
        print(f"   Loss: {loss.item():.6f}")
        
        # Test mark_step for synchronization
        xm.mark_step()
        print("✅ XLA mark_step successful")
        
        return True
    except Exception as e:
        print(f"❌ Optimizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mixed_precision():
    """Test mixed precision (BF16) with PyTorch 2.7"""
    print("\n🔥 Testing Mixed Precision (BF16)")
    print("=" * 60)
    
    try:
        import torch
        import torch_xla.core.xla_model as xm
        
        device = xm.xla_device()
        
        # Check if BF16 is supported
        print(f"✅ BF16 environment variable: {os.environ.get('XLA_USE_BF16', 'Not set')}")
        
        # Test autocast (should work with XLA)
        with torch.autocast(device_type='xla', dtype=torch.bfloat16):
            x = torch.randn(8, 256, device=device)
            y = torch.matmul(x, x.transpose(-2, -1))
            result = torch.softmax(y, dim=-1)
        
        print(f"✅ Autocast with BF16 successful")
        print(f"   Input dtype: {x.dtype}")
        print(f"   Result dtype: {result.dtype}")
        print(f"   Shape: {result.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Mixed precision test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all PyTorch 2.7 / XLA 2.7 compatibility tests"""
    print("🚀 PyTorch 2.7.1 / XLA 2.7.0 TPU Compatibility Test")
    print("=" * 80)
    
    tests = [
        ("Version Check", test_pytorch_xla_versions),
        ("Model on TPU", test_model_on_tpu),
        ("Data Loading", test_data_loading_apis),
        ("Optimizer APIs", test_optimizer_apis),
        ("Mixed Precision", test_mixed_precision),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            start_time = time.time()
            result = test_func()
            elapsed = time.time() - start_time
            results[name] = result
            print(f"   ⏱️ {elapsed:.1f}s")
        except Exception as e:
            print(f"❌ Test {name} crashed: {e}")
            results[name] = False
    
    print("\n" + "=" * 80)
    print("📊 PyTorch 2.7 Compatibility Summary")
    print("=" * 80)
    
    passed = 0
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All PyTorch 2.7 compatibility tests passed!")
        print("    Ready for TPU smoke test with updated APIs.")
        return 0
    elif passed >= 3:
        print("⚠️  Some tests failed, but core PyTorch 2.7 functionality works.")
        print("    Smoke test should work with minor issues.")
        return 1
    else:
        print("❌ Critical PyTorch 2.7 compatibility issues detected.")
        print("    Fix version issues before running smoke test.")
        return 2

if __name__ == "__main__":
    exit(main())