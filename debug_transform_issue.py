#!/usr/bin/env python3
"""Debug script to identify the transform pipeline issue."""

import torch
from data_setup.datasets import create_file_lists
from data_setup.gcs_dicom_reader import GCSDICOMReader
from data_setup.transforms import LoadDICOMd
from monai.transforms import Spacingd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_dicom_reader():
    """Test raw DICOM reading."""
    print("=" * 50)
    print("1. TESTING RAW DICOM READING")
    print("=" * 50)
    
    file_list = create_file_lists(
        'gs://layne-tpu-code-sync/OCTdata/OCTdata/manifest.tsv',
        'gs://layne-tpu-code-sync/OCTdata/OCTdata',
        'single_domain'
    )
    
    reader = GCSDICOMReader()
    test_file = file_list[0]
    print(f"Testing file: {test_file}")
    
    result = reader.read_dicom_volume(test_file)
    if result:
        print(f"✅ Raw DICOM read successful")
        print(f"   Shape: {result['pixel_array'].shape}")
        print(f"   Spacing: {result['spacing']}")
        print(f"   Metadata keys: {list(result['metadata'].keys())}")
        return result
    else:
        print("❌ Raw DICOM read failed")
        return None

def debug_dataset_sample(dicom_data):
    """Test dataset sample creation."""
    print("=" * 50)
    print("2. TESTING DATASET SAMPLE CREATION")
    print("=" * 50)
    
    if not dicom_data:
        print("❌ No DICOM data to test")
        return None
    
    # Simulate what OCTDICOMDataset.__getitem__ does
    pixel_array = dicom_data['pixel_array']
    spacing = dicom_data['spacing']
    metadata = dicom_data['metadata']
    
    # Convert to tensor and add channel dimension
    image = torch.from_numpy(pixel_array).unsqueeze(0).float()
    
    sample = {
        'image': image,
        'spacing': spacing,
        'meta': {
            **metadata,
            'original_shape': pixel_array.shape,
            'filepath': 'test_file',
            'target_spacing': (0.05, 0.02, 0.02),
            'target_size': (64, 384, 384)
        }
    }
    
    print(f"✅ Sample created")
    print(f"   Image shape: {sample['image'].shape}")
    print(f"   Image type: {type(sample['image'])}")
    print(f"   Spacing: {sample['spacing']}")
    print(f"   Meta keys: {list(sample['meta'].keys())}")
    
    return sample

def debug_load_transform(sample):
    """Test LoadDICOMd transform."""
    print("=" * 50)
    print("3. TESTING LoadDICOMd TRANSFORM")
    print("=" * 50)
    
    if not sample:
        print("❌ No sample to test")
        return None
    
    try:
        transform = LoadDICOMd(keys=['image'])
        result = transform(sample)
        
        print(f"✅ LoadDICOMd successful")
        print(f"   Image type: {type(result['image'])}")
        print(f"   Image shape: {result['image'].shape}")
        
        if hasattr(result['image'], 'meta'):
            print(f"   MetaTensor meta: {result['image'].meta}")
        
        return result
        
    except Exception as e:
        print(f"❌ LoadDICOMd failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def debug_spacing_transform(sample):
    """Test Spacingd transform."""
    print("=" * 50)
    print("4. TESTING Spacingd TRANSFORM")  
    print("=" * 50)
    
    if not sample:
        print("❌ No sample to test")
        return None
    
    try:
        transform = Spacingd(
            keys=['image'],
            pixdim=(0.05, 0.02, 0.02),
            mode='bilinear'
        )
        result = transform(sample)
        
        print(f"✅ Spacingd successful")
        print(f"   Image shape: {result['image'].shape}")
        
        return result
        
    except Exception as e:
        print(f"❌ Spacingd failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main debug function."""
    print("🔍 DEBUG: Transform Pipeline Issue")
    print("=" * 80)
    
    try:
        # Step 1: Test raw DICOM reading
        dicom_data = debug_dicom_reader()
        
        # Step 2: Test dataset sample creation
        sample = debug_dataset_sample(dicom_data)
        
        # Step 3: Test LoadDICOMd transform
        loaded_sample = debug_load_transform(sample)
        
        # Step 4: Test Spacingd transform
        spaced_sample = debug_spacing_transform(loaded_sample)
        
        print("=" * 80)
        print("🎉 DEBUG COMPLETE")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())