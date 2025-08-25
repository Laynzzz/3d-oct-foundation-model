#!/usr/bin/env python3
"""
Test script to verify Backblaze B2 connection and list bucket contents.
Requires environment variables to be set (see .env.example).
"""

import os
import sys
from typing import List

try:
    import s3fs
except ImportError:
    print("Error: s3fs not installed. Run: pip install s3fs")
    sys.exit(1)


def test_b2_connection() -> bool:
    """Test connection to Backblaze B2 bucket."""
    
    # Check required environment variables
    required_vars = [
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY', 
        'S3_ENDPOINT_URL'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"Error: Missing environment variables: {missing_vars}")
        print("Please copy .env.example to .env and fill in your B2 credentials")
        return False
    
    try:
        # Initialize S3 filesystem with B2 endpoint
        fs = s3fs.S3FileSystem(
            key=os.getenv('AWS_ACCESS_KEY_ID'),
            secret=os.getenv('AWS_SECRET_ACCESS_KEY'),
            endpoint_url=os.getenv('S3_ENDPOINT_URL'),
            use_ssl=True
        )
        
        bucket_name = "eye-dataset"
        print(f"Testing connection to bucket: {bucket_name}")
        
        # Test bucket access by listing contents
        try:
            contents = fs.ls(bucket_name, detail=False)[:10]  # Limit to first 10 items
            print(f"✅ Successfully connected to {bucket_name}")
            print(f"Found {len(contents)} items (showing first 10):")
            for item in contents:
                print(f"  - {item}")
            
            # Look for fine-tuning data specifically
            fine_tuning_prefix = f"{bucket_name}/fine-tuneing-data"
            try:
                ft_contents = fs.ls(fine_tuning_prefix, detail=False)[:5]
                print(f"\n✅ Found fine-tuning data at {fine_tuning_prefix}:")
                for item in ft_contents:
                    print(f"  - {item}")
            except Exception as e:
                print(f"\n⚠️  Could not access {fine_tuning_prefix}: {e}")
                
            return True
            
        except Exception as e:
            print(f"❌ Failed to access bucket {bucket_name}: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Failed to create S3 filesystem: {e}")
        return False


if __name__ == "__main__":
    print("Testing Backblaze B2 connection...")
    success = test_b2_connection()
    sys.exit(0 if success else 1)