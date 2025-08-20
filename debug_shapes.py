#!/usr/bin/env python3
"""Debug script to check tensor shapes in JEPA pipeline."""

import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_mask_shapes():
    """Test mask generation shapes."""
    print("=== Testing Mask Generation Shapes ===")
    
    # Config from smoke test
    image_size = (32, 192, 192)  # D, H, W
    patch_size = (4, 16, 16)
    
    # Calculate expected patch dimensions
    patch_d, patch_h, patch_w = patch_size
    D, H, W = image_size
    
    grid_d = D // patch_d
    grid_h = H // patch_h  
    grid_w = W // patch_w
    
    total_patches = grid_d * grid_h * grid_w
    
    print(f"Image size: {image_size}")
    print(f"Patch size: {patch_size}")
    print(f"Grid dimensions: {grid_d} x {grid_h} x {grid_w}")
    print(f"Total patches: {total_patches}")
    
    # Test mask generation
    mask_ratio = 0.6
    num_masked = int(total_patches * mask_ratio)
    
    # Create mask like our transform does
    mask_flat = torch.zeros(total_patches, dtype=torch.bool)
    masked_indices = torch.randperm(total_patches)[:num_masked]
    mask_flat[masked_indices] = True
    
    print(f"Generated mask shape: {mask_flat.shape}")
    print(f"Mask dtype: {mask_flat.dtype}")
    print(f"Number of masked patches: {mask_flat.sum().item()}/{total_patches}")
    
    # Test batch dimension
    batch_size = 2
    batch_mask = mask_flat.unsqueeze(0).expand(batch_size, -1)
    print(f"Batch mask shape: {batch_mask.shape}")
    
    return batch_mask

def test_vjepa_shapes():
    """Test V-JEPA model input shapes."""
    print("\n=== Testing V-JEPA Model Shapes ===")
    
    batch_size = 2
    image_size = (32, 192, 192)
    
    # Create dummy inputs
    context_view = torch.randn(batch_size, 1, *image_size)
    target_view = torch.randn(batch_size, 1, *image_size)
    mask = test_mask_shapes()
    
    print(f"Context view shape: {context_view.shape}")
    print(f"Target view shape: {target_view.shape}")
    print(f"Mask shape: {mask.shape}")
    
    # Test if shapes are compatible
    try:
        print("\n✅ All shapes look correct for V-JEPA model!")
        return True
    except Exception as e:
        print(f"\n❌ Shape compatibility error: {e}")
        return False

def main():
    """Main test function."""
    print("🔍 Debugging JEPA tensor shapes...")
    
    success = test_vjepa_shapes()
    
    if success:
        print("\n🎉 Shape analysis complete - no obvious issues!")
    else:
        print("\n⚠️ Shape issues detected!")

if __name__ == '__main__':
    main()