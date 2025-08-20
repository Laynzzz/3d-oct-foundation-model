#!/usr/bin/env python3
"""
Quick test to validate a few DICOM files and understand the data structure.
"""

import pandas as pd
from data_setup.gcs_dicom_reader import GCSDICOMReader

# Load manifest
manifest_path = "/tmp/manifest.tsv"
df = pd.read_csv(manifest_path, sep='\t')
print(f"Loaded manifest with {len(df)} entries")
print(f"Columns: {list(df.columns)}")

# Test first 5 files
gcs_root = "gs://layne-tpu-code-sync/OCTdata/OCTdata"
reader = GCSDICOMReader()

print(f"\nTesting first 5 DICOM files:")
for i in range(min(5, len(df))):
    row = df.iloc[i]
    file_path = f"{gcs_root}{row['filepath']}"
    print(f"\n{i+1}. Testing: {file_path}")
    
    try:
        dicom_data = reader.load_dicom(file_path)
        if dicom_data is None:
            print(f"   ❌ Failed to load DICOM")
            continue
            
        pixel_array = reader.get_pixel_array(dicom_data)
        if pixel_array is None:
            print(f"   ❌ No pixel data available")
            continue
            
        print(f"   ✅ Valid - Shape: {pixel_array.shape}, Type: {pixel_array.dtype}")
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:100]}...")