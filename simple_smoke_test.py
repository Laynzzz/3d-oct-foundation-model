#!/usr/bin/env python3
"""Simple smoke test without torchrun to bypass TPU permission issues"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def main():
    """Run simple smoke test without distributed training"""
    print("🚀 Simple TPU Smoke Test (No Distributed Training)")
    print("=" * 60)
    
    try:
        # Basic imports
        import torch
        import torch_xla.core.xla_model as xm
        print(f"✅ PyTorch {torch.__version__}, XLA {torch_xla.__version__}")
        
        # Set environment to simulate single worker
        os.environ['LOCAL_RANK'] = '0'
        os.environ['RANK'] = '0' 
        os.environ['WORLD_SIZE'] = '1'
        
        # Get device
        device = xm.xla_device()
        print(f"✅ Device: {device}")
        
        # Import model
        from models.vjepa_3d import VJEPA3D
        print("✅ Model imported")
        
        # Create minimal model
        model = VJEPA3D(
            img_size=(16, 64, 64),
            patch_size=(4, 16, 16),
            embed_dim=192,  # Small
            depth=6,        # Small
            num_heads=6,    # Small
            ema_momentum=0.996
        ).to(device)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Model created: {param_count:,} parameters")
        
        # Test forward pass
        batch_size = 1
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
        
        # Sync
        xm.mark_step()
        elapsed = time.time() - start_time
        
        print(f"✅ Forward pass successful!")
        print(f"   Loss: {loss.item():.6f}")
        print(f"   Predictions: {predictions.shape}")
        print(f"   Targets: {targets.shape}")
        print(f"   Time: {elapsed:.3f}s")
        
        # Test backward pass
        model.train()
        loss, _, _ = model(context, target, mask)
        loss.backward()
        xm.mark_step()
        
        # Check gradients
        grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5
        
        print(f"✅ Backward pass successful, grad norm: {grad_norm:.6f}")
        
        print("\n🎉 Simple smoke test PASSED!")
        print("   Model can run forward and backward passes on TPU")
        print("   Ready for distributed training once TPU permissions are resolved")
        
        return True
        
    except Exception as e:
        print(f"❌ Simple smoke test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)