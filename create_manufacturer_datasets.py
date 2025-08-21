#!/usr/bin/env python3
"""
Create manufacturer-specific datasets based on success rates observed.
Heidelberg had 100% success rate, so extract all Heidelberg files first.
"""

import pandas as pd
import subprocess
from pathlib import Path

def create_heidelberg_dataset():
    """Extract all Heidelberg files since they had 100% success rate."""
    
    print("Loading original manifest...")
    manifest_path = "/tmp/original_manifest.tsv"
    df = pd.read_csv(manifest_path, sep='\t')
    print(f"Total files in original manifest: {len(df)}")
    
    # Filter by manufacturer
    heidelberg_df = df[df['manufacturer'].str.contains('Heidelberg', case=False, na=False)].copy()
    print(f"Heidelberg files found: {len(heidelberg_df)}")
    
    if len(heidelberg_df) > 0:
        print("\nHeidelberg file breakdown:")
        print(heidelberg_df.groupby(['manufacturer', 'manufacturers_model_name', 'anatomic_region']).size())
        
        # Save Heidelberg-only manifest
        output_path = "/tmp/manifest_heidelberg.tsv"
        heidelberg_df.to_csv(output_path, sep='\t', index=False)
        
        # Upload to GCS
        gcs_output = "gs://layne-tpu-code-sync/OCTdata/OCTdata/manifest_heidelberg.tsv"
        subprocess.run(['gsutil', 'cp', output_path, gcs_output], check=True)
        
        print(f"\n✅ Created Heidelberg dataset:")
        print(f"   Files: {len(heidelberg_df)}")
        print(f"   Saved to: {gcs_output}")
        
        return len(heidelberg_df)
    else:
        print("❌ No Heidelberg files found!")
        return 0

def create_manufacturer_breakdown():
    """Analyze manufacturer distribution in the full dataset."""
    
    manifest_path = "/tmp/original_manifest.tsv" 
    df = pd.read_csv(manifest_path, sep='\t')
    
    print(f"\n📊 Full Dataset Manufacturer Analysis:")
    print(f"Total files: {len(df)}")
    
    manufacturer_counts = df['manufacturer'].value_counts()
    print(f"\nManufacturer distribution:")
    for manufacturer, count in manufacturer_counts.items():
        percentage = count/len(df)*100
        print(f"  {manufacturer}: {count:,} files ({percentage:.1f}%)")
    
    # Detailed breakdown by model and region
    print(f"\nDetailed breakdown:")
    detailed = df.groupby(['manufacturer', 'manufacturers_model_name', 'anatomic_region']).size().sort_values(ascending=False)
    for (mfg, model, region), count in detailed.head(10).items():
        print(f"  {mfg} {model} {region}: {count:,} files")
    
    return manufacturer_counts

def create_mixed_reliable_dataset():
    """Create a mixed dataset with manufacturers that showed high success rates."""
    
    manifest_path = "/tmp/original_manifest.tsv"
    df = pd.read_csv(manifest_path, sep='\t')
    
    # Based on our testing: Heidelberg=100%, Topcon=good, Zeiss=good
    reliable_manufacturers = ['Heidelberg', 'Topcon', 'Zeiss'] 
    
    reliable_df = df[df['manufacturer'].isin(reliable_manufacturers)].copy()
    
    print(f"\n🔄 Creating mixed reliable dataset:")
    print(f"Including manufacturers: {reliable_manufacturers}")
    print(f"Files selected: {len(reliable_df)} out of {len(df)}")
    
    # Show breakdown
    print(f"\nReliable dataset breakdown:")
    breakdown = reliable_df.groupby('manufacturer').size()
    for mfg, count in breakdown.items():
        print(f"  {mfg}: {count:,} files")
    
    # Save mixed reliable manifest
    output_path = "/tmp/manifest_reliable.tsv"
    reliable_df.to_csv(output_path, sep='\t', index=False)
    
    # Upload to GCS
    gcs_output = "gs://layne-tpu-code-sync/OCTdata/OCTdata/manifest_reliable.tsv"
    subprocess.run(['gsutil', 'cp', output_path, gcs_output], check=True)
    
    print(f"\n✅ Created reliable mixed dataset:")
    print(f"   Files: {len(reliable_df)}")
    print(f"   Saved to: {gcs_output}")
    
    return len(reliable_df)

def main():
    """Main function to create manufacturer-based datasets."""
    
    print("🔧 Creating Manufacturer-Specific Datasets")
    print("=" * 50)
    
    # Analyze full dataset
    manufacturer_counts = create_manufacturer_breakdown()
    
    # Create Heidelberg-only dataset (100% success rate)
    heidelberg_count = create_heidelberg_dataset()
    
    # Create mixed reliable dataset
    reliable_count = create_mixed_reliable_dataset()
    
    print(f"\n📋 Summary:")
    print(f"   Original dataset: {sum(manufacturer_counts):,} files")
    print(f"   Heidelberg-only: {heidelberg_count:,} files")
    print(f"   Mixed reliable: {reliable_count:,} files")
    print(f"   Minimal test: 20 files")
    
    print(f"\n🎯 Next steps:")
    print(f"   1. Test training with Heidelberg-only dataset")
    print(f"   2. Validate more files from mixed reliable dataset")
    print(f"   3. Scale up to larger validated dataset")

if __name__ == "__main__":
    main()