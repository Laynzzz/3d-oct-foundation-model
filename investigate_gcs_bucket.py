#!/usr/bin/env python3
"""
GCS Bucket Investigation Script
Analyzes the OCT data bucket to understand file availability and corruption issues.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import json
from collections import defaultdict, Counter
import urllib.parse
import re

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    from google.cloud import storage
    from google.api_core import exceptions as gcs_exceptions
    import pandas as pd
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Please install: pip install google-cloud-storage pandas")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gcs_investigation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GCSBucketInvestigator:
    """Investigates GCS bucket for data integrity issues."""
    
    def __init__(self, bucket_name: str, project_id: str):
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.client = None
        self.bucket = None
        self.results = {
            'total_files': 0,
            'accessible_files': 0,
            'missing_files': 0,
            'corrupted_paths': [],
            'valid_files': [],
            'directory_structure': defaultdict(int),
            'file_extensions': Counter(),
            'size_distribution': [],
            'path_encoding_issues': [],
            'permission_errors': []
        }
        
    def initialize_client(self):
        """Initialize GCS client and bucket."""
        try:
            self.client = storage.Client(project=self.project_id)
            self.bucket = self.client.bucket(self.bucket_name)
            logger.info(f"Successfully connected to bucket: {self.bucket_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
            return False
    
    def analyze_manifest_file(self, manifest_path: str) -> List[str]:
        """Load and analyze the manifest file."""
        logger.info(f"Loading manifest file: {manifest_path}")
        
        if not os.path.exists(manifest_path):
            logger.error(f"Manifest file not found: {manifest_path}")
            return []
        
        try:
            with open(manifest_path, 'r') as f:
                lines = f.readlines()
            
            file_paths = []
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract GCS path from manifest line
                    if '\t' in line:
                        parts = line.split('\t')
                        gcs_path = parts[0] if parts[0].startswith('gs://') else None
                    else:
                        gcs_path = line if line.startswith('gs://') else None
                    
                    if gcs_path:
                        file_paths.append(gcs_path)
                    else:
                        logger.warning(f"Invalid line {line_num} in manifest: {line}")
            
            logger.info(f"Loaded {len(file_paths)} file paths from manifest")
            self.results['total_files'] = len(file_paths)
            return file_paths
            
        except Exception as e:
            logger.error(f"Error reading manifest file: {e}")
            return []
    
    def check_path_encoding(self, gcs_path: str) -> Tuple[bool, str]:
        """Check for path encoding issues."""
        try:
            # Remove gs://bucket_name/ prefix
            path_parts = gcs_path.replace(f'gs://{self.bucket_name}/', '')
            
            # Check for double encoding or malformed URLs
            if '%2F' in path_parts or '%2f' in path_parts:
                self.results['path_encoding_issues'].append({
                    'original': gcs_path,
                    'issue': 'Double URL encoding detected',
                    'decoded': urllib.parse.unquote(path_parts)
                })
                return False, "Double URL encoding"
            
            # Check for other encoding issues
            if '%' in path_parts and not path_parts.replace('%20', ' ').replace('%2B', '+').isascii():
                self.results['path_encoding_issues'].append({
                    'original': gcs_path,
                    'issue': 'Invalid URL encoding',
                    'decoded': urllib.parse.unquote(path_parts)
                })
                return False, "Invalid URL encoding"
            
            return True, "OK"
            
        except Exception as e:
            return False, f"Encoding check error: {e}"
    
    def check_file_exists(self, gcs_path: str) -> Tuple[bool, Dict]:
        """Check if a file exists in GCS and get its metadata."""
        try:
            # Extract blob path (remove gs://bucket_name/)
            blob_path = gcs_path.replace(f'gs://{self.bucket_name}/', '')
            
            # Get blob
            blob = self.bucket.blob(blob_path)
            
            # Check if exists
            if blob.exists():
                # Get metadata
                blob.reload()
                metadata = {
                    'size': blob.size,
                    'created': blob.time_created.isoformat() if blob.time_created else None,
                    'updated': blob.updated.isoformat() if blob.updated else None,
                    'content_type': blob.content_type,
                    'etag': blob.etag
                }
                return True, metadata
            else:
                return False, {'error': 'File not found'}
                
        except gcs_exceptions.Forbidden as e:
            self.results['permission_errors'].append({
                'path': gcs_path,
                'error': f'Permission denied: {e}'
            })
            return False, {'error': f'Permission denied: {e}'}
        except Exception as e:
            return False, {'error': f'Check failed: {e}'}
    
    def analyze_directory_structure(self, file_paths: List[str]):
        """Analyze the directory structure from file paths."""
        logger.info("Analyzing directory structure...")
        
        for gcs_path in file_paths:
            # Extract directory path
            path_without_bucket = gcs_path.replace(f'gs://{self.bucket_name}/', '')
            path_parts = path_without_bucket.split('/')
            
            # Count files per directory level
            for i in range(1, len(path_parts)):
                dir_path = '/'.join(path_parts[:i])
                self.results['directory_structure'][dir_path] += 1
            
            # File extension
            if '.' in path_parts[-1]:
                ext = path_parts[-1].split('.')[-1].lower()
                self.results['file_extensions'][ext] += 1
    
    def investigate_files(self, file_paths: List[str], sample_size: Optional[int] = None):
        """Investigate a sample of files for existence and accessibility."""
        if sample_size and len(file_paths) > sample_size:
            import random
            file_paths = random.sample(file_paths, sample_size)
            logger.info(f"Sampling {sample_size} files from {len(file_paths)} total")
        
        logger.info(f"Investigating {len(file_paths)} files...")
        
        for i, gcs_path in enumerate(file_paths):
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{len(file_paths)} files checked")
            
            # Check path encoding
            encoding_ok, encoding_issue = self.check_path_encoding(gcs_path)
            
            # Check file existence
            exists, metadata = self.check_file_exists(gcs_path)
            
            if exists:
                self.results['accessible_files'] += 1
                self.results['valid_files'].append({
                    'path': gcs_path,
                    'metadata': metadata
                })
                if metadata.get('size'):
                    self.results['size_distribution'].append(metadata['size'])
            else:
                self.results['missing_files'] += 1
                self.results['corrupted_paths'].append({
                    'path': gcs_path,
                    'encoding_ok': encoding_ok,
                    'encoding_issue': encoding_issue,
                    'error': metadata.get('error', 'Unknown error')
                })
    
    def list_actual_bucket_contents(self, prefix: str = "OCTdata/", max_files: int = 1000):
        """List what actually exists in the bucket."""
        logger.info(f"Listing actual bucket contents with prefix: {prefix}")
        
        try:
            blobs = self.client.list_blobs(
                self.bucket_name, 
                prefix=prefix, 
                max_results=max_files
            )
            
            actual_files = []
            for blob in blobs:
                actual_files.append({
                    'name': blob.name,
                    'size': blob.size,
                    'created': blob.time_created.isoformat() if blob.time_created else None,
                    'content_type': blob.content_type
                })
            
            logger.info(f"Found {len(actual_files)} actual files in bucket")
            return actual_files
            
        except Exception as e:
            logger.error(f"Failed to list bucket contents: {e}")
            return []
    
    def generate_report(self) -> Dict:
        """Generate comprehensive investigation report."""
        report = {
            'summary': {
                'total_files_in_manifest': self.results['total_files'],
                'accessible_files': self.results['accessible_files'],
                'missing_files': self.results['missing_files'],
                'accessibility_rate': (self.results['accessible_files'] / max(self.results['total_files'], 1)) * 100,
                'path_encoding_issues': len(self.results['path_encoding_issues']),
                'permission_errors': len(self.results['permission_errors'])
            },
            'file_extensions': dict(self.results['file_extensions']),
            'directory_structure': dict(self.results['directory_structure']),
            'size_stats': self._calculate_size_stats(),
            'top_missing_directories': self._analyze_missing_directories(),
            'sample_corrupted_paths': self.results['corrupted_paths'][:10],
            'sample_valid_files': self.results['valid_files'][:5],
            'path_encoding_issues': self.results['path_encoding_issues'][:10],
            'permission_errors': self.results['permission_errors'][:10]
        }
        
        return report
    
    def _calculate_size_stats(self) -> Dict:
        """Calculate file size statistics."""
        if not self.results['size_distribution']:
            return {}
        
        sizes = self.results['size_distribution']
        return {
            'count': len(sizes),
            'total_size_bytes': sum(sizes),
            'avg_size_bytes': sum(sizes) / len(sizes),
            'min_size_bytes': min(sizes),
            'max_size_bytes': max(sizes)
        }
    
    def _analyze_missing_directories(self) -> List[Dict]:
        """Analyze which directories have the most missing files."""
        missing_dirs = defaultdict(int)
        
        for item in self.results['corrupted_paths']:
            path = item['path']
            path_without_bucket = path.replace(f'gs://{self.bucket_name}/', '')
            dir_path = '/'.join(path_without_bucket.split('/')[:-1])
            missing_dirs[dir_path] += 1
        
        # Sort by count
        sorted_dirs = sorted(missing_dirs.items(), key=lambda x: x[1], reverse=True)
        
        return [{'directory': dir_path, 'missing_count': count} for dir_path, count in sorted_dirs[:10]]
    
    def save_results(self, output_dir: str = "gcs_investigation_results"):
        """Save investigation results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main report
        report = self.generate_report()
        with open(f"{output_dir}/investigation_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save valid files list
        if self.results['valid_files']:
            valid_paths = [item['path'] for item in self.results['valid_files']]
            with open(f"{output_dir}/valid_files.txt", 'w') as f:
                f.write('\n'.join(valid_paths))
        
        # Save corrupted paths
        if self.results['corrupted_paths']:
            with open(f"{output_dir}/corrupted_paths.json", 'w') as f:
                json.dump(self.results['corrupted_paths'], f, indent=2)
        
        # Save path encoding issues
        if self.results['path_encoding_issues']:
            with open(f"{output_dir}/path_encoding_issues.json", 'w') as f:
                json.dump(self.results['path_encoding_issues'], f, indent=2)
        
        logger.info(f"Results saved to {output_dir}/")


def main():
    """Main investigation function."""
    # Configuration
    BUCKET_NAME = "layne-tpu-code-sync"
    PROJECT_ID = "d-oct-foundational-model"
    MANIFEST_PATH = "data_manifests/topcon_triton_train.txt"
    SAMPLE_SIZE = 500  # Check a sample of files for speed
    
    logger.info("🔍 Starting GCS Bucket Investigation")
    logger.info(f"Bucket: {BUCKET_NAME}")
    logger.info(f"Project: {PROJECT_ID}")
    logger.info(f"Manifest: {MANIFEST_PATH}")
    
    # Initialize investigator
    investigator = GCSBucketInvestigator(BUCKET_NAME, PROJECT_ID)
    
    # Connect to GCS
    if not investigator.initialize_client():
        logger.error("Failed to connect to GCS. Exiting.")
        return
    
    # Load manifest
    file_paths = investigator.analyze_manifest_file(MANIFEST_PATH)
    if not file_paths:
        logger.error("No files found in manifest. Exiting.")
        return
    
    # Analyze directory structure
    investigator.analyze_directory_structure(file_paths)
    
    # Investigate file accessibility
    investigator.investigate_files(file_paths, sample_size=SAMPLE_SIZE)
    
    # List actual bucket contents
    actual_files = investigator.list_actual_bucket_contents()
    
    # Generate and save report
    report = investigator.generate_report()
    investigator.save_results()
    
    # Print summary
    print("\n" + "="*60)
    print("🔍 GCS BUCKET INVESTIGATION SUMMARY")
    print("="*60)
    print(f"Total files in manifest: {report['summary']['total_files_in_manifest']}")
    print(f"Accessible files: {report['summary']['accessible_files']}")
    print(f"Missing files: {report['summary']['missing_files']}")
    print(f"Accessibility rate: {report['summary']['accessibility_rate']:.1f}%")
    print(f"Path encoding issues: {report['summary']['path_encoding_issues']}")
    print(f"Permission errors: {report['summary']['permission_errors']}")
    
    if actual_files:
        print(f"Actual files found in bucket: {len(actual_files)}")
    
    print("\n📊 File Extensions:")
    for ext, count in report['file_extensions'].items():
        print(f"  .{ext}: {count} files")
    
    print("\n📁 Top Missing Directories:")
    for item in report['top_missing_directories'][:5]:
        print(f"  {item['directory']}: {item['missing_count']} missing files")
    
    if report['sample_corrupted_paths']:
        print(f"\n❌ Sample Corrupted Paths:")
        for item in report['sample_corrupted_paths'][:3]:
            print(f"  {item['path']}")
            print(f"    Error: {item['error']}")
    
    print(f"\n📄 Full results saved to: gcs_investigation_results/")
    print("="*60)


if __name__ == "__main__":
    main()
