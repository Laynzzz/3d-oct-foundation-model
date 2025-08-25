"""
Backblaze B2 storage utilities for fine-tuning data access.
Provides S3-compatible interface to B2 bucket.
"""

import os
import time
from typing import Optional, List
from functools import lru_cache
import s3fs
import logging

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_s3fs() -> s3fs.S3FileSystem:
    """
    Create cached S3 filesystem instance for Backblaze B2.
    Uses environment variables for configuration.
    """
    required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'S3_ENDPOINT_URL']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {missing_vars}")
    
    return s3fs.S3FileSystem(
        key=os.getenv('AWS_ACCESS_KEY_ID'),
        secret=os.getenv('AWS_SECRET_ACCESS_KEY'),
        endpoint_url=os.getenv('S3_ENDPOINT_URL'),
        use_ssl=True
    )


def list_bucket_contents(bucket_name: str, prefix: str = "", max_keys: int = 1000) -> List[str]:
    """
    List contents of B2 bucket with optional prefix filtering.
    
    Args:
        bucket_name: Name of the bucket
        prefix: Optional prefix to filter results
        max_keys: Maximum number of keys to return
        
    Returns:
        List of object keys
    """
    fs = get_s3fs()
    
    try:
        full_prefix = f"{bucket_name}/{prefix}" if prefix else bucket_name
        contents = fs.ls(full_prefix, detail=False)
        return contents[:max_keys]
    except Exception as e:
        logger.error(f"Failed to list bucket contents: {e}")
        raise


def file_exists(bucket_name: str, key: str) -> bool:
    """Check if a file exists in the B2 bucket."""
    fs = get_s3fs()
    full_path = f"{bucket_name}/{key}"
    
    try:
        return fs.exists(full_path)
    except Exception as e:
        logger.error(f"Error checking file existence for {full_path}: {e}")
        return False


def read_file_with_retry(bucket_name: str, key: str, max_retries: int = 3) -> bytes:
    """
    Read file from B2 bucket with retry logic.
    
    Args:
        bucket_name: Name of the bucket
        key: Object key to read
        max_retries: Maximum number of retry attempts
        
    Returns:
        File contents as bytes
    """
    fs = get_s3fs()
    full_path = f"{bucket_name}/{key}"
    
    for attempt in range(max_retries + 1):
        try:
            with fs.open(full_path, 'rb') as f:
                return f.read()
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Failed to read {full_path} after {max_retries} retries: {e}")
                raise
            else:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.warning(f"Retry {attempt + 1}/{max_retries} for {full_path} in {wait_time}s: {e}")
                time.sleep(wait_time)


def get_cache_path(cache_dir: str, bucket_name: str, key: str) -> str:
    """Generate local cache path for a B2 object."""
    import hashlib
    
    # Create a safe filename from the key
    safe_key = key.replace('/', '_').replace('\\', '_')
    key_hash = hashlib.md5(key.encode()).hexdigest()[:8]
    filename = f"{safe_key}_{key_hash}"
    
    return os.path.join(cache_dir, bucket_name, filename)


def read_with_cache(bucket_name: str, key: str, cache_dir: Optional[str] = None) -> bytes:
    """
    Read file from B2 bucket with optional local caching.
    
    Args:
        bucket_name: Name of the bucket
        key: Object key to read
        cache_dir: Optional local cache directory
        
    Returns:
        File contents as bytes
    """
    # If no cache directory, read directly
    if not cache_dir:
        return read_file_with_retry(bucket_name, key)
    
    # Check cache first
    cache_path = get_cache_path(cache_dir, bucket_name, key)
    
    if os.path.exists(cache_path):
        logger.debug(f"Reading from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            return f.read()
    
    # Cache miss - download from B2
    logger.debug(f"Cache miss, downloading: {bucket_name}/{key}")
    data = read_file_with_retry(bucket_name, key)
    
    # Save to cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        f.write(data)
    
    logger.debug(f"Cached to: {cache_path}")
    return data


def test_connection(bucket_name: str = "eye-dataset") -> bool:
    """
    Test connection to B2 bucket.
    
    Args:
        bucket_name: Name of the bucket to test
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        contents = list_bucket_contents(bucket_name, max_keys=1)
        logger.info(f"Successfully connected to bucket: {bucket_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to bucket {bucket_name}: {e}")
        return False