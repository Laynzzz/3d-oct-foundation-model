#!/usr/bin/env python3
"""
Create a minimal working dataset by testing files one by one and keeping only those that work.
"""

import pandas as pd
from data_setup.gcs_dicom_reader import GCSDICOMReader
import time

def test_minimal_files():
    """Test files and create a minimal working dataset."""
    
    # Load manifest
    manifest_path = "/tmp/manifest.tsv"
    df = pd.read_csv(manifest_path, sep='\t')
    print(f"Loaded manifest with {len(df)} entries")
    
    # Initialize reader
    reader = GCSDICOMReader(use_cache=False)  # Disable cache for testing
    gcs_root = "gs://layne-tpu-code-sync/OCTdata/OCTdata"
    
    valid_files = []
    tested_count = 0
    max_to_find = 20  # Find at least 20 working files for smoke test
    max_to_test = 100  # Don't test more than 100 files
    
    print(f"Testing up to {max_to_test} files to find {max_to_find} working ones...")
    
    for i, row in df.iterrows():
        if tested_count >= max_to_test or len(valid_files) >= max_to_find:
            break
            
        file_path = f"{gcs_root}{row['filepath']}"
        tested_count += 1
        
        try:
            print(f"{tested_count:3d}. Testing: {row['filepath'][-50:]}")
            
            # Try to read the DICOM
            result = reader.read_dicom_volume(file_path)
            
            if result is None:
                print(f"     ❌ Failed to load")
                continue
                
            pixel_array = result.get('pixel_array')
            if pixel_array is None:
                print(f"     ❌ No pixel data")
                continue
                
            if pixel_array.size == 0:
                print(f"     ❌ Empty pixel array")
                continue
                
            if len(pixel_array.shape) < 3:
                print(f"     ❌ Invalid shape: {pixel_array.shape}")
                continue
                
            print(f"     ✅ Valid - Shape: {pixel_array.shape}")
            valid_files.append(row)
            
        except Exception as e:
            print(f"     ❌ Error: {str(e)[:50]}...")
            continue
    
    print(f"\nFound {len(valid_files)} valid files out of {tested_count} tested")
    
    if len(valid_files) == 0:
        print("❌ No valid files found! This suggests a fundamental issue with the data or reader.")
        return
    
    # Create minimal manifest
    valid_df = pd.DataFrame(valid_files)
    
    # Save locally first
    output_path = "/tmp/manifest_minimal.tsv"
    valid_df.to_csv(output_path, sep='\t', index=False)
    
    # Upload to GCS
    import subprocess
    gcs_output = "gs://layne-tpu-code-sync/OCTdata/OCTdata/manifest_minimal.tsv"
    subprocess.run(['gsutil', 'cp', output_path, gcs_output], check=True)
    
    print(f"\n✅ Created minimal dataset:")
    print(f"   Valid files: {len(valid_files)}")
    print(f"   Success rate: {len(valid_files)/tested_count*100:.1f}%")
    print(f"   Saved to: {gcs_output}")
    
    # Show some examples
    print(f"\nExample valid files:")
    for i, row in enumerate(valid_files[:3]):
        print(f"   {i+1}. {row['filepath']}")

if __name__ == "__main__":
    test_minimal_files()