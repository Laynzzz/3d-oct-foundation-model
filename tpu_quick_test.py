#!/usr/bin/env python3
"""
Quick TPU validation script - run this before the full smoke test
Tests basic TPU functionality and imports
"""

import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_tpu_basic():
    """Test basic TPU functionality"""
    print("🔧 Testing TPU Basic Functionality")
    print("=" * 50)
    
    try:
        import torch
        import torch_xla
        import torch_xla.core.xla_model as xm
        
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ XLA available: {torch_xla._XLAC._xla_runtime_is_initialized()}")
        
        device_count = xm.xrt.device_count()
        print(f"✅ TPU device count: {device_count}")
        
        if device_count == 0:
            print("❌ No TPU devices found!")
            return False
            
        device = xm.xla_device()
        print(f"✅ TPU device: {device}")
        
        # Test basic tensor operations
        x = torch.randn(5, 5, device=device)
        y = torch.matmul(x, x)
        result = y.cpu().numpy()
        print(f"✅ Basic tensor ops work: {result.shape}")
        
        return True
    except Exception as e:
        print(f"❌ TPU test failed: {e}")
        return False

def test_model_imports():
    """Test model imports"""
    print("\n🧠 Testing Model Imports")
    print("=" * 50)
    
    try:
        from models.vjepa_3d import VJEPA3D
        from data_setup.datasets import OCTDICOMDataset
        from data_setup.transforms import create_pretraining_transforms
        from utils.config_parser import load_config
        
        print("✅ All core imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gcs_access():
    """Test GCS access"""
    print("\n☁️  Testing GCS Access")
    print("=" * 50)
    
    try:
        import gcsfs
        fs = gcsfs.GCSFileSystem()
        
        # Test bucket access
        bucket_path = "gs://layne-tpu-code-sync/OCTdata/OCTdata"
        try:
            files = fs.ls(bucket_path)
            print(f"✅ GCS bucket accessible: {len(files)} items")
        except Exception as e:
            print(f"❌ GCS bucket access failed: {e}")
            return False
        
        # Test manifest file
        manifest_path = f"{bucket_path}/manifest.tsv"
        try:
            with fs.open(manifest_path, 'r') as f:
                header = f.readline()
            print(f"✅ Manifest accessible: {header[:50]}...")
        except Exception as e:
            print(f"❌ Manifest access failed: {e}")
            return False
            
        # Test DICOM directory
        dicom_path = f"{bucket_path}/retinal_oct/structural_oct/topcon_triton"
        try:
            dicom_files = fs.ls(dicom_path)
            print(f"✅ DICOM files found: {len(dicom_files)} files")
        except Exception as e:
            print(f"❌ DICOM access failed: {e}")
            return False
            
        return True
    except Exception as e:
        print(f"❌ GCS test failed: {e}")
        return False

def test_tiny_model():
    """Test tiny model creation and forward pass on TPU"""
    print("\n🚀 Testing Tiny Model on TPU")
    print("=" * 50)
    
    try:
        import torch
        import torch_xla.core.xla_model as xm
        from models.vjepa_3d import VJEPA3D
        
        device = xm.xla_device()
        
        # Create tiny model
        model = VJEPA3D(
            img_size=(16, 64, 64),    # Very small
            patch_size=(4, 16, 16),
            embed_dim=192,            # Small
            depth=2,                  # Very shallow
            num_heads=4,              # Few heads
            ema_momentum=0.996
        ).to(device)
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"✅ Tiny model created: {param_count:,} parameters")
        
        # Test forward pass
        batch_size = 1
        context = torch.randn(batch_size, 1, 16, 64, 64, device=device)
        target = torch.randn(batch_size, 1, 16, 64, 64, device=device)
        
        # Create proper mask
        num_patches = (16//4) * (64//16) * (64//16)  # 4 * 4 * 4 = 64
        mask = torch.randint(0, 2, (batch_size, num_patches), device=device).bool()
        
        print(f"   Input shapes: {context.shape}, {target.shape}, {mask.shape}")
        
        with torch.no_grad():
            loss, predictions, targets = model(context, target, mask)
        
        print(f"✅ Forward pass successful!")
        print(f"   Loss: {loss.item():.6f}")
        print(f"   Predictions: {predictions.shape}")
        print(f"   Targets: {targets.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config_loading():
    """Test config loading"""
    print("\n⚙️  Testing Config Loading")
    print("=" * 50)
    
    try:
        from utils.config_parser import load_config
        config = load_config("configs/smoke_test.yaml")
        
        print(f"✅ Config loaded: {config.experiment_name}")
        print(f"   Batch size: {config.global_batch_size}")
        print(f"   Max steps: {config.max_steps}")
        print(f"   Image size: {config.image_size}")
        
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def main():
    """Run all TPU validation tests"""
    print("🚀 TPU Quick Validation Test")
    print("=" * 60)
    
    tests = [
        ("TPU Basic", test_tpu_basic),
        ("Model Imports", test_model_imports),
        ("GCS Access", test_gcs_access),
        ("Config Loading", test_config_loading),
        ("Tiny Model", test_tiny_model),
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
    
    print("\n" + "=" * 60)
    print("📊 Validation Summary")
    print("=" * 60)
    
    passed = 0
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {name:15} : {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Ready for TPU smoke test.")
        print("\nNext: Run 'bash run_tpu.sh configs/smoke_test.yaml'")
        return 0
    elif passed >= 3:
        print("⚠️  Some tests failed, but core functionality works.")
        print("   You can still try the smoke test, but expect issues.")
        return 1
    else:
        print("❌ Multiple critical failures. Fix issues before smoke test.")
        return 2

if __name__ == "__main__":
    exit(main())