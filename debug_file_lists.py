#!/usr/bin/env python3
"""
Debug File Lists Script
Investigates what files the training code actually selects.
"""

import os
import sys
import yaml
from pathlib import Path
from google.cloud import storage

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data_setup.datasets import create_file_lists

def main():
    print('🔍 Debugging File List Selection')
    print('-' * 50)
    
    # Load config
    config_path = 'configs/pretrain_vjepa_single_domain.yaml'
    with open(config_path, 'r') as f:
        config_text = f.read()
    
    print('Raw config content (data section):')
    lines = config_text.split('\n')
    in_data_section = False
    for line in lines:
        if line.strip().startswith('# data'):
            in_data_section = True
        elif line.strip().startswith('#') and in_data_section and not line.strip().startswith('# data'):
            break
        elif in_data_section:
            print(f'   {line}')
    
    # Parse config and expand variables manually
    config = yaml.safe_load(config_text)
    gcs_root = config['gcs_root']
    manifest_path = config['manifest_path'].replace('${gcs_root}', gcs_root)
    list_strategy = config['list_strategy']
    
    print(f'\nExpanded paths:')
    print(f'   GCS root: {gcs_root}')
    print(f'   Manifest path: {manifest_path}')
    print(f'   List strategy: {list_strategy}')
    
    # Create file lists using the expanded path
    try:
        print(f'\nCreating file lists...')
        file_lists = create_file_lists(
            manifest_path,
            gcs_root, 
            list_strategy=list_strategy
        )
        
        train_files = file_lists['train']
        val_files = file_lists['val']
        
        print(f'✅ Successfully created file lists:')
        print(f'   Training files: {len(train_files)}')
        print(f'   Validation files: {len(val_files)}')
        
        print(f'\n📄 First 10 training files:')
        for i, file_path in enumerate(train_files[:10]):
            print(f'   {i+1}: {file_path}')
        
        # Check if these files actually exist
        print(f'\n🔍 Checking if training files exist in GCS...')
        client = storage.Client(project='d-oct-foundational-model')
        bucket = client.bucket('layne-tpu-code-sync')
        
        missing_count = 0
        existing_count = 0
        
        for i, gcs_path in enumerate(train_files[:20]):  # Check first 20
            blob_path = gcs_path.replace('gs://layne-tpu-code-sync/', '')
            blob = bucket.blob(blob_path)
            exists = blob.exists()
            
            if exists:
                existing_count += 1
                if existing_count <= 3:  # Show first 3 existing
                    print(f'   ✅ {gcs_path}')
            else:
                missing_count += 1
                if missing_count <= 5:  # Show first 5 missing
                    print(f'   ❌ {gcs_path}')
        
        print(f'\n📊 Summary of first 20 files:')
        print(f'   Existing: {existing_count}/20')
        print(f'   Missing: {missing_count}/20')
        print(f'   Availability: {(existing_count/20)*100:.1f}%')
        
        if missing_count > 0:
            print(f'\n🔍 Analyzing missing file patterns...')
            missing_files = []
            for gcs_path in train_files[:100]:  # Check more files
                blob_path = gcs_path.replace('gs://layne-tpu-code-sync/', '')
                blob = bucket.blob(blob_path)
                if not blob.exists():
                    missing_files.append(gcs_path)
                if len(missing_files) >= 10:
                    break
            
            print(f'Common patterns in missing files:')
            manufacturers = {}
            for path in missing_files:
                if 'topcon_triton' in path:
                    manufacturers['topcon_triton'] = manufacturers.get('topcon_triton', 0) + 1
                elif 'heidelberg_spectralis' in path:
                    manufacturers['heidelberg_spectralis'] = manufacturers.get('heidelberg_spectralis', 0) + 1
                elif 'topcon_maestro2' in path:
                    manufacturers['topcon_maestro2'] = manufacturers.get('topcon_maestro2', 0) + 1
                elif 'zeiss_cirrus' in path:
                    manufacturers['zeiss_cirrus'] = manufacturers.get('zeiss_cirrus', 0) + 1
            
            for mfg, count in manufacturers.items():
                print(f'   {mfg}: {count} missing files')
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
