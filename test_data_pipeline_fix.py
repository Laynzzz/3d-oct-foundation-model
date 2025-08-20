#!/usr/bin/env python3
"""Test data pipeline with improved DICOM validation."""

import os
import sys
import logging
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_setup.datasets import OCTDICOMDataset, create_file_lists, collate_fn
from data_setup.transforms import create_validation_transforms


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def test_single_file_loading(logger):
    """Test loading a single DICOM file."""
    from data_setup.gcs_dicom_reader import GCSDICOMReader
    
    # Test with a known file
    test_paths = [
        "gs://layne-tpu-code-sync/OCTdata/OCTdata/retinal_oct/structural_oct/topcon_triton/1001/1001_20160920_131842_OD_sn2016927_oct_tslo_512x496.dcm",
        "gs://layne-tpu-code-sync/OCTdata/OCTdata/retinal_oct/structural_oct/topcon_triton/1002/1002_20160920_132033_OS_sn2016927_oct_tslo_512x496.dcm",
        "gs://layne-tpu-code-sync/OCTdata/OCTdata/retinal_oct/structural_oct/topcon_triton/1003/1003_20160920_132233_OD_sn2016927_oct_tslo_512x496.dcm"
    ]
    
    reader = GCSDICOMReader(use_cache=False)
    
    for i, path in enumerate(test_paths):
        logger.info(f"Testing file {i+1}/{len(test_paths)}: {path}")
        try:
            result = reader.read_dicom_volume(path)
            if result is not None:
                logger.info(f"✅ File {i+1} loaded successfully - shape: {result['pixel_array'].shape}")
                logger.info(f"   Spacing: {result['spacing']}")
                logger.info(f"   Metadata keys: {list(result['metadata'].keys())}")
            else:
                logger.warning(f"⚠️ File {i+1} returned None (likely no pixel data)")
        except Exception as e:
            logger.error(f"❌ File {i+1} failed to load: {e}")
    
    logger.info(f"Total skipped files: {reader.get_skipped_files_count()}")


def test_dataset_loading(logger):
    """Test dataset loading with improved validation."""
    
    # Configuration
    manifest_path = "gs://layne-tpu-code-sync/OCTdata/OCTdata/manifest.tsv"
    gcs_root = "gs://layne-tpu-code-sync/OCTdata/OCTdata"
    
    logger.info("Creating file lists...")
    try:
        all_files = create_file_lists(
            manifest_path=manifest_path,
            gcs_root=gcs_root,
            list_strategy="single_domain",  # Focus on topcon_triton
            participant_range=(1001, 1020)  # Small subset for testing
        )
        
        logger.info(f"Found {len(all_files)} files in range 1001-1020")
        
        if len(all_files) == 0:
            logger.error("No files found in participant range - check manifest and filtering")
            return
        
        # Take first 10 files for testing
        test_files = all_files[:10]
        logger.info(f"Testing with first 10 files: {len(test_files)} files")
        
        # Create validation transforms (no augmentation)
        transforms = create_validation_transforms(
            target_spacing=(0.05, 0.02, 0.02),
            image_size=(64, 384, 384)
        )
        
        # Create dataset
        dataset = OCTDICOMDataset(
            manifest_path=manifest_path,
            gcs_root=gcs_root,
            file_list=test_files,
            transforms=transforms,
            use_cache=False
        )
        
        logger.info(f"Created dataset with {len(dataset)} samples")
        
        # Test individual samples
        logger.info("Testing individual samples...")
        valid_samples = 0
        for i in range(min(5, len(dataset))):
            try:
                sample = dataset[i]
                if sample is not None:
                    valid_samples += 1
                    logger.info(f"✅ Sample {i}: shape {sample['image'].shape}, spacing {sample['spacing']}")
                else:
                    logger.warning(f"⚠️ Sample {i}: returned None")
            except Exception as e:
                logger.error(f"❌ Sample {i}: failed with {e}")
        
        logger.info(f"Valid samples: {valid_samples}/{min(5, len(dataset))}")
        
        # Test DataLoader with custom collate function
        logger.info("Testing DataLoader with collate function...")
        
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,  # Single-threaded to avoid multiprocessing issues
            collate_fn=collate_fn,
            drop_last=False
        )
        
        batch_count = 0
        valid_batch_count = 0
        
        for batch_idx, batch in enumerate(dataloader):
            batch_count += 1
            if batch is not None:
                valid_batch_count += 1
                logger.info(f"✅ Batch {batch_idx}: {batch['image'].shape[0]} samples, "
                           f"image shape {batch['image'].shape}")
            else:
                logger.warning(f"⚠️ Batch {batch_idx}: returned None (all samples failed)")
            
            # Test only first 3 batches
            if batch_idx >= 2:
                break
        
        logger.info(f"DataLoader test complete: {valid_batch_count}/{batch_count} valid batches")
        
        if valid_batch_count > 0:
            logger.info("🎉 Data pipeline working with improved validation!")
        else:
            logger.error("❌ All batches failed - data pipeline still has issues")
            
    except Exception as e:
        logger.error(f"Dataset loading test failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())


def main():
    """Main test function."""
    logger = setup_logging()
    
    logger.info("🔧 Testing OCT Data Pipeline with Improved DICOM Validation")
    logger.info("=" * 60)
    
    # Test 1: Single file loading
    logger.info("\n📁 Test 1: Single File Loading")
    logger.info("-" * 40)
    test_single_file_loading(logger)
    
    # Test 2: Dataset and DataLoader
    logger.info("\n📊 Test 2: Dataset and DataLoader")
    logger.info("-" * 40)
    test_dataset_loading(logger)
    
    logger.info("\n✅ Testing complete!")


if __name__ == '__main__':
    main()