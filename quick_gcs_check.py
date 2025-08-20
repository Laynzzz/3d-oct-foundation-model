#!/usr/bin/env python3
"""
Quick GCS Bucket Check
Simple script to check file availability issues in the OCT data bucket.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict
import json
from collections import Counter
import urllib.parse

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    from google.cloud import storage
    from google.api_core import exceptions as gcs_exceptions
except ImportError as e:
    print(f"Missing google-cloud-storage: {e}")
    print("Installing...")
    os.system("pip install google-cloud-storage")
    from google.cloud import storage
    from google.api_core import exceptions as gcs_exceptions

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_gcs_files(bucket_name: str, project_id: str, manifest_path: str, max_check: int = 100):
    """Quick check of GCS file availability."""
    
    print(f"🔍 Quick GCS Check")
    print(f"Bucket: {bucket_name}")
    print(f"Project: {project_id}")
    print(f"Manifest: {manifest_path}")
    print(f"Max files to check: {max_check}")
    print("-" * 50)
    
    # Load manifest
    if not os.path.exists(manifest_path):
        print(f"❌ Manifest file not found: {manifest_path}")
        return
    
    with open(manifest_path, 'r') as f:
        lines = f.readlines()
    
    file_paths = []
    for line in lines:
        line = line.strip()
        if line and line.startswith('gs://'):
            file_paths.append(line)
    
    print(f"📄 Found {len(file_paths)} files in manifest")
    
    if len(file_paths) > max_check:
        file_paths = file_paths[:max_check]
        print(f"🔬 Checking first {max_check} files...")
    
    # Initialize GCS client
    try:
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        print(f"✅ Connected to GCS bucket")
    except Exception as e:
        print(f"❌ Failed to connect to GCS: {e}")
        return
    
    # Check files
    results = {
        'total_checked': 0,
        'accessible': 0,
        'missing': 0,
        'permission_errors': 0,
        'encoding_issues': 0,
        'sample_missing': [],
        'sample_accessible': []
    }
    
    for i, gcs_path in enumerate(file_paths):
        if i % 20 == 0:
            print(f"Progress: {i}/{len(file_paths)}")
        
        results['total_checked'] += 1
        
        try:
            # Extract blob path
            blob_path = gcs_path.replace(f'gs://{bucket_name}/', '')
            
            # Check for encoding issues
            if '%2F' in blob_path or '%2f' in blob_path:
                results['encoding_issues'] += 1
                results['sample_missing'].append({
                    'path': gcs_path,
                    'issue': 'Double URL encoding detected'
                })
                continue
            
            # Check if file exists
            blob = bucket.blob(blob_path)
            
            if blob.exists():
                results['accessible'] += 1
                if len(results['sample_accessible']) < 3:
                    blob.reload()
                    results['sample_accessible'].append({
                        'path': gcs_path,
                        'size': blob.size,
                        'content_type': blob.content_type
                    })
            else:
                results['missing'] += 1
                if len(results['sample_missing']) < 10:
                    results['sample_missing'].append({
                        'path': gcs_path,
                        'issue': 'File not found in bucket'
                    })
                    
        except gcs_exceptions.Forbidden:
            results['permission_errors'] += 1
        except Exception as e:
            results['missing'] += 1
            if len(results['sample_missing']) < 10:
                results['sample_missing'].append({
                    'path': gcs_path,
                    'issue': f'Error: {str(e)[:100]}'
                })
    
    # Print results
    print("\n" + "="*60)
    print("📊 QUICK GCS CHECK RESULTS")
    print("="*60)
    print(f"Total files checked: {results['total_checked']}")
    print(f"Accessible files: {results['accessible']}")
    print(f"Missing files: {results['missing']}")
    print(f"Permission errors: {results['permission_errors']}")
    print(f"Encoding issues: {results['encoding_issues']}")
    
    accessibility_rate = (results['accessible'] / max(results['total_checked'], 1)) * 100
    print(f"Accessibility rate: {accessibility_rate:.1f}%")
    
    if results['sample_accessible']:
        print(f"\n✅ Sample Accessible Files:")
        for item in results['sample_accessible']:
            print(f"  📁 {item['path']}")
            print(f"     Size: {item['size']} bytes, Type: {item['content_type']}")
    
    if results['sample_missing']:
        print(f"\n❌ Sample Missing Files:")
        for item in results['sample_missing'][:5]:
            print(f"  📁 {item['path']}")
            print(f"     Issue: {item['issue']}")
    
    # List actual bucket contents
    print(f"\n🔍 Checking actual bucket contents...")
    try:
        blobs = client.list_blobs(bucket_name, prefix="OCTdata/", max_results=50)
        actual_files = list(blobs)
        print(f"Found {len(actual_files)} actual files in bucket with OCTdata/ prefix")
        
        if actual_files:
            print("Sample actual files:")
            for blob in actual_files[:5]:
                print(f"  📄 {blob.name} ({blob.size} bytes)")
        else:
            print("⚠️  No files found with OCTdata/ prefix!")
            
            # Try without prefix
            print("Checking bucket root...")
            blobs = client.list_blobs(bucket_name, max_results=20)
            root_files = list(blobs)
            print(f"Found {len(root_files)} files in bucket root")
            for blob in root_files[:10]:
                print(f"  📄 {blob.name}")
                
    except Exception as e:
        print(f"❌ Error listing bucket contents: {e}")
    
    print("="*60)
    
    # Save results
    with open('quick_gcs_check_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("Results saved to: quick_gcs_check_results.json")


def main():
    """Main function."""
    BUCKET_NAME = "layne-tpu-code-sync"
    PROJECT_ID = "d-oct-foundational-model"
    MANIFEST_PATH = "data_manifests/topcon_triton_train.txt"
    
    check_gcs_files(BUCKET_NAME, PROJECT_ID, MANIFEST_PATH, max_check=100)


if __name__ == "__main__":
    main()
