#!/usr/bin/env python3
"""
Smoke test for OCT V-JEPA2 model - tests basic functionality locally
"""

import os
import sys
import torch
import traceback
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        from utils.config_parser import load_config
        from data_setup.datasets import OCTDICOMDataset, create_file_lists
        from data_setup.transforms import create_pretraining_transforms, get_val_transforms
        from models.vjepa_3d import VJEPA3D
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config_loading():
    """Test config loading"""
    print("\nTesting config loading...")
    try:
        from utils.config_parser import load_config
        config = load_config("configs/smoke_test.yaml")
        print(f"✅ Config loaded: {config.experiment_name}")
        return config
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        return None

def test_data_pipeline():
    """Test data pipeline without GCS (will likely fail but shows structure)"""
    print("\nTesting data pipeline structure...")
    try:
        from data_setup.datasets import OCTDICOMDataset, create_file_lists
        from data_setup.transforms import create_pretraining_transforms
        
        # This will likely fail due to GCS access, but tests the structure
        config = {
            'gcs_root': 'gs://layne-tpu-code-sync/OCTdata/OCTdata',
            'manifest_path': 'gs://layne-tpu-code-sync/OCTdata/OCTdata/manifest.tsv',
            'target_spacing': [0.05, 0.02, 0.02],
            'image_size': [32, 192, 192],
            'patch_size': [4, 16, 16],
            'mask_ratio': 0.6
        }
        
        transforms = create_pretraining_transforms(
            target_spacing=config['target_spacing'],
            image_size=config['image_size'],
            patch_size=config['patch_size'],
            mask_ratio=config['mask_ratio']
        )
        print("✅ Transform pipeline created")
        
        # Test file list creation (will fail with GCS but shows structure)
        try:
            file_lists = create_file_lists(
                config['manifest_path'], 
                config['gcs_root'],
                list_strategy='single_domain'
            )
            print("✅ File list creation attempted")
        except Exception as e:
            print(f"⚠️  File list creation failed (expected without GCS): {e}")
        
        return True
    except Exception as e:
        print(f"❌ Data pipeline error: {e}")
        traceback.print_exc()
        return False

def test_model():
    """Test model creation and forward pass"""
    print("\nTesting model...")
    try:
        from models.vjepa_3d import VJEPA3D
        
        config = {
            'embed_dim': 384,  # Smaller for test
            'depth': 6,        # Smaller for test
            'num_heads': 6,
            'patch_size': (4, 16, 16),
            'img_size': (32, 192, 192),
            'ema_momentum': 0.996
        }
        
        model = VJEPA3D(**config)
        print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test forward pass with dummy data
        batch_size = 2
        C, D, H, W = 1, 32, 192, 192
        
        dummy_context = torch.randn(batch_size, C, D, H, W)
        dummy_target = torch.randn(batch_size, C, D, H, W)
        # Mask should match the number of patches: D//4 * H//16 * W//16 = 8 * 12 * 12 = 1152
        num_patches = (D//4) * (H//16) * (W//16)
        dummy_mask = torch.randint(0, 2, (batch_size, num_patches)).bool()
        
        with torch.no_grad():
            loss, predictions, targets = model(dummy_context, dummy_target, dummy_mask)
        
        print(f"✅ Forward pass successful - Loss: {loss.item():.4f}")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Targets shape: {targets.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Model error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all smoke tests"""
    print("🚀 Starting OCT V-JEPA2 Smoke Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_loading, 
        test_data_pipeline,
        test_model
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Smoke Test Results:")
    print(f"   Passed: {sum(1 for r in results if r)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! Ready for training.")
        return 0
    else:
        print("⚠️  Some tests failed. Check implementation.")
        return 1

if __name__ == "__main__":
    exit(main())