#!/usr/bin/env python3
"""Debug script to identify the exact path corruption issue."""

import sys
import logging
from utils.config_parser import load_config
from data_setup.manifest_parser import ManifestParser
from data_setup.datasets import create_file_lists

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_config_loading():
    """Test config loading and template variable expansion."""
    print("=" * 50)
    print("1. TESTING CONFIG LOADING")
    print("=" * 50)
    
    config = load_config('configs/pretrain_vjepa_single_domain.yaml')
    
    print(f"gcs_root: '{config.gcs_root}'")
    print(f"manifest_path: '{config.manifest_path}'")
    print(f"gcs_root type: {type(config.gcs_root)}")
    print(f"manifest_path type: {type(config.manifest_path)}")
    
    return config

def debug_manifest_parser(config):
    """Test manifest parser initialization and path construction."""
    print("=" * 50)
    print("2. TESTING MANIFEST PARSER")
    print("=" * 50)
    
    print(f"Creating ManifestParser with:")
    print(f"  manifest_path: '{config.manifest_path}'")
    print(f"  gcs_root: '{config.gcs_root}'")
    
    # Initialize parser
    parser = ManifestParser(config.manifest_path, config.gcs_root)
    print(f"Parser gcs_root after init: '{parser.gcs_root}'")
    
    # Load manifest
    print("Loading manifest...")
    df = parser.load_manifest()
    
    # Check first few rows for path construction
    print("First 3 rows of manifest after processing:")
    print(df[['filepath', 'gcs_path', 'device']].head(3))
    
    # Check for any corrupted paths
    sample_paths = df['gcs_path'].head(5).tolist()
    print(f"\nFirst 5 constructed GCS paths:")
    for i, path in enumerate(sample_paths):
        print(f"  {i+1}: '{path}'")
        if 'gs://' in path[5:]:  # Check for duplicate gs:// after the first one
            print(f"      ⚠️  CORRUPTION DETECTED: Multiple 'gs://' in path!")
        if config.gcs_root in path[len(config.gcs_root):]:  # Check for duplicate gcs_root
            print(f"      ⚠️  CORRUPTION DETECTED: Duplicate gcs_root in path!")
    
    return parser

def debug_file_list_creation(config):
    """Test file list creation function."""
    print("=" * 50)
    print("3. TESTING FILE LIST CREATION")
    print("=" * 50)
    
    print("Creating file list with single_domain strategy...")
    try:
        file_list = create_file_lists(
            manifest_path=config.manifest_path,
            gcs_root=config.gcs_root,
            list_strategy='single_domain'
        )
        
        print(f"File list length: {len(file_list)}")
        print("First 3 files in list:")
        for i, path in enumerate(file_list[:3]):
            print(f"  {i+1}: '{path}'")
            if 'gs://' in path[5:]:  # Check for duplicate gs:// after the first one
                print(f"      ⚠️  CORRUPTION DETECTED: Multiple 'gs://' in path!")
            if config.gcs_root in path[len(config.gcs_root):]:  # Check for duplicate gcs_root
                print(f"      ⚠️  CORRUPTION DETECTED: Duplicate gcs_root in path!")
        
        return file_list
        
    except Exception as e:
        print(f"❌ Error in file list creation: {e}")
        import traceback
        traceback.print_exc()
        return []

def debug_gcs_path_construction():
    """Test manual GCS path construction to identify the issue."""
    print("=" * 50)
    print("4. MANUAL PATH CONSTRUCTION TEST")
    print("=" * 50)
    
    # Simulate the values from config
    gcs_root = "gs://layne-tpu-code-sync/OCTdata/OCTdata"
    manifest_path = f"{gcs_root}/manifest.tsv"
    
    # Test filepath from manifest
    sample_filepath = "/retinal_oct/structural_oct/heidelberg_spectralis/1001/1001_spectralis_onh_rc_hr_oct_l_1.3.6.1.4.1.33437.11.4.7587979.98316546453556.22400.4.1.dcm"
    
    print(f"gcs_root: '{gcs_root}'")
    print(f"sample_filepath: '{sample_filepath}'")
    
    # Test current method (problematic)
    current_method = f"{gcs_root}{sample_filepath}"
    print(f"Current method result: '{current_method}'")
    
    # Test corrected method
    gcs_root_clean = gcs_root.rstrip('/')
    sample_filepath_clean = sample_filepath.lstrip('/')
    corrected_method = f"{gcs_root_clean}/{sample_filepath_clean}"
    print(f"Corrected method result: '{corrected_method}'")
    
    if current_method != corrected_method:
        print("⚠️  PATH CONSTRUCTION DIFFERS - this might be the issue!")
    else:
        print("✅ Path construction is consistent")

def main():
    """Main debug function."""
    print("🔍 DEBUG: Path Corruption Investigation")
    print("=" * 80)
    
    try:
        # Step 1: Test config loading
        config = debug_config_loading()
        
        # Step 2: Test manifest parser
        parser = debug_manifest_parser(config)
        
        # Step 3: Test file list creation  
        file_list = debug_file_list_creation(config)
        
        # Step 4: Manual path construction test
        debug_gcs_path_construction()
        
        print("=" * 80)
        print("🎉 DEBUG COMPLETE")
        print("Check the output above for any corruption warnings (⚠️)")
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())