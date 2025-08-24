#!/usr/bin/env python3
"""
Smoke test runner for the OCT fine-tuning pipeline.
Validates all components are working correctly before running full training.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from finetuning.utils.checks import run_pipeline_smoke_test

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run smoke tests with command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run smoke tests for OCT fine-tuning pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--labels",
        default="fine-tuneing-data/participants.tsv",
        help="Path to participants.tsv file"
    )
    
    parser.add_argument(
        "--checkpoints-dir",
        default="/Users/layne/Mac/Acdamic/UCInspire/checkpoints",
        help="Directory containing V-JEPA2 checkpoints"
    )
    
    parser.add_argument(
        "--test-b2",
        action="store_true",
        help="Test B2 connection (requires credentials in environment)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only essential tests (skip optional components)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Resolve paths
    labels_path = Path(args.labels)
    if not labels_path.is_absolute():
        labels_path = Path(__file__).parent / labels_path
    
    checkpoints_dir = Path(args.checkpoints_dir)
    
    # Find checkpoint files
    checkpoint_names = [
        "best_checkpoint_multi_domain.pt",
        "best_checkpoint_single_domain_01.pt", 
        "best_checkpoint_single_domain_02.pt"
    ]
    
    checkpoint_paths = []
    for name in checkpoint_names:
        checkpoint_path = checkpoints_dir / name
        if checkpoint_path.exists():
            checkpoint_paths.append(str(checkpoint_path))
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
    
    if not checkpoint_paths:
        logger.error(f"No checkpoints found in {checkpoints_dir}")
        logger.error("Expected files:")
        for name in checkpoint_names:
            logger.error(f"  - {checkpoints_dir / name}")
        sys.exit(1)
    
    # Check labels file
    if not labels_path.exists():
        logger.error(f"Labels file not found: {labels_path}")
        sys.exit(1)
    
    # Display test configuration
    logger.info("🧪 OCT Fine-Tuning Pipeline Smoke Tests")
    logger.info("=" * 50)
    logger.info(f"Labels file: {labels_path}")
    logger.info(f"Checkpoints directory: {checkpoints_dir}")
    logger.info(f"Found checkpoints: {len(checkpoint_paths)}")
    for cp in checkpoint_paths:
        logger.info(f"  - {Path(cp).name}")
    logger.info(f"Test B2 connection: {args.test_b2}")
    logger.info(f"Quick mode: {args.quick}")
    logger.info("=" * 50)
    
    # Check environment variables if testing B2
    if args.test_b2:
        required_env_vars = [
            'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY', 
            'S3_ENDPOINT_URL'
        ]
        
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        if missing_vars:
            logger.error("B2 testing requested but missing environment variables:")
            for var in missing_vars:
                logger.error(f"  - {var}")
            logger.error("Please set these variables or copy .env.example to .env and source it")
            sys.exit(1)
    
    # Run smoke tests
    try:
        logger.info("🚀 Starting smoke tests...")
        
        success = run_pipeline_smoke_test(
            labels_path=str(labels_path),
            checkpoint_paths=checkpoint_paths,
            test_b2_connection=args.test_b2
        )
        
        if success:
            logger.info("🎉 All smoke tests PASSED! Pipeline is ready for training.")
            print("\n" + "=" * 60)
            print("✅ SMOKE TESTS PASSED")
            print("=" * 60)
            print("The fine-tuning pipeline is ready!")
            print("\nNext steps:")
            print("1. Set up B2 credentials (copy .env.example to .env)")
            print("2. Run linear probe:")
            print("   python -m finetuning.train.run --config configs/cls_linear_probe.yaml")
            print("3. Run fine-tuning:")
            print("   python -m finetuning.train.run --config configs/cls_finetune.yaml") 
            print("4. Run multi-checkpoint sweep:")
            print("   python -m finetuning.train.run --config configs/sweep_checkpoints.yaml -m")
            print("=" * 60)
            
        else:
            logger.error("💥 Some smoke tests FAILED! Please fix issues before training.")
            print("\n" + "=" * 60)
            print("❌ SMOKE TESTS FAILED")
            print("=" * 60)
            print("Please review the test results above and fix any issues.")
            print("Common fixes:")
            print("1. Install missing dependencies: pip install -r requirements.txt")
            print("2. Check file paths and permissions")
            print("3. Set up B2 credentials if testing B2 connection")
            print("4. Ensure checkpoint files exist and are valid")
            print("=" * 60)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Smoke tests interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"Smoke tests failed with exception: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()