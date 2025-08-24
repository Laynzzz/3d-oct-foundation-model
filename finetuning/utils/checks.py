"""
Validation checks module for fine-tuning pipeline.
Smoke tests and validation utilities for data, models, and training.
"""

import torch
import os
import sys
from typing import Optional, Dict, List, Tuple, Any
import logging
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from finetuning.storage.b2 import test_connection, list_bucket_contents
from finetuning.data.labels import load_labels, process_labels
from finetuning.data.locator import get_default_locator
from finetuning.data.dataset import OCTVolumeDataset, DebugDataset, create_dataloader, create_debug_dataloader
from finetuning.data.transforms import VJepa2Transforms, validate_transform_output
from finetuning.models.encoder_loader import load_vjepa2_encoder, validate_encoder_output
from finetuning.models.model import create_model_from_checkpoint

logger = logging.getLogger(__name__)


class ValidationResult:
    """Container for validation results."""
    
    def __init__(self, name: str, passed: bool, message: str, details: Optional[Dict] = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status}: {self.name} - {self.message}"


class ValidationSuite:
    """Test suite for validating fine-tuning pipeline components."""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
    
    def add_result(self, name: str, passed: bool, message: str, details: Optional[Dict] = None):
        """Add validation result."""
        result = ValidationResult(name, passed, message, details)
        self.results.append(result)
        logger.info(str(result))
        return result
    
    def check_b2_connection(self, bucket_name: str = "eye-dataset") -> ValidationResult:
        """Test B2 bucket connectivity."""
        try:
            success = test_connection(bucket_name)
            if success:
                # Try to list a few items
                contents = list_bucket_contents(bucket_name, max_keys=5)
                return self.add_result(
                    "B2 Connection",
                    True,
                    f"Successfully connected to {bucket_name}",
                    {"bucket": bucket_name, "sample_contents": contents[:3]}
                )
            else:
                return self.add_result(
                    "B2 Connection",
                    False,
                    f"Failed to connect to bucket {bucket_name}",
                    {"bucket": bucket_name}
                )
        except Exception as e:
            return self.add_result(
                "B2 Connection",
                False,
                f"Exception during B2 test: {str(e)}",
                {"error": str(e)}
            )
    
    def check_labels_loading(self, labels_path: str) -> ValidationResult:
        """Test labels TSV loading and processing."""
        try:
            if not os.path.exists(labels_path):
                return self.add_result(
                    "Labels Loading",
                    False,
                    f"Labels file not found: {labels_path}",
                    {"path": labels_path}
                )
            
            # Load labels
            df = load_labels(labels_path)
            
            # Process labels
            train_df, val_df, test_df, class_to_idx, class_weights = process_labels(labels_path)
            
            details = {
                "total_participants": len(df),
                "with_oct": len(df[df.get('retinal_oct', False) == True]) if 'retinal_oct' in df.columns else "Unknown",
                "train_size": len(train_df),
                "val_size": len(val_df), 
                "test_size": len(test_df),
                "num_classes": len(class_to_idx),
                "class_mapping": class_to_idx,
                "class_weights": {int(k): float(v) for k, v in class_weights.items()}
            }
            
            return self.add_result(
                "Labels Loading",
                True,
                f"Successfully processed {len(df)} participants",
                details
            )
            
        except Exception as e:
            return self.add_result(
                "Labels Loading",
                False,
                f"Failed to load labels: {str(e)}",
                {"error": str(e), "path": labels_path}
            )
    
    def check_oct_locator(self, sample_participants: List[str] = None) -> ValidationResult:
        """Test OCT data locator functionality."""
        try:
            locator = get_default_locator()
            
            # Get available participants
            available = locator.get_available_participants()
            
            if not available:
                return self.add_result(
                    "OCT Locator",
                    False,
                    "No OCT data found by locator",
                    {"available_count": 0}
                )
            
            # Test key resolution for sample participants
            sample_participants = sample_participants or available[:5]
            resolved_keys = []
            
            for pid in sample_participants:
                key = locator.resolve_oct_key(str(pid))
                if key:
                    resolved_keys.append({"participant_id": pid, "key": key})
            
            success_rate = len(resolved_keys) / len(sample_participants) if sample_participants else 0
            
            return self.add_result(
                "OCT Locator",
                success_rate > 0.5,  # At least 50% success rate
                f"Resolved {len(resolved_keys)}/{len(sample_participants)} participant keys",
                {
                    "total_available": len(available),
                    "tested_participants": len(sample_participants),
                    "resolved_keys": len(resolved_keys),
                    "success_rate": success_rate,
                    "sample_resolutions": resolved_keys[:3]
                }
            )
            
        except Exception as e:
            return self.add_result(
                "OCT Locator",
                False,
                f"OCT locator test failed: {str(e)}",
                {"error": str(e)}
            )
    
    def check_dummy_dataset(self, labels_path: str, batch_size: int = 2) -> ValidationResult:
        """Test dataset creation with dummy data."""
        try:
            # Process labels
            train_df, val_df, test_df, class_to_idx, class_weights = process_labels(labels_path)
            
            if len(train_df) == 0:
                return self.add_result(
                    "Dummy Dataset",
                    False,
                    "No training data available",
                    {"train_size": 0}
                )
            
            # Create transforms
            transforms = VJepa2Transforms(target_shape=(64, 384, 384), augment=False)
            
            # Create debug dataset (with dummy data)
            debug_loader = create_debug_dataloader(
                train_df.head(8),  # Use first 8 samples
                batch_size=batch_size,
                transforms=transforms,
                shuffle=False
            )
            
            # Test loading a batch
            batch = next(iter(debug_loader))
            volumes, labels, participant_ids = batch
            
            # Validate batch
            expected_volume_shape = (batch_size, 1, 64, 384, 384)
            expected_labels_shape = (batch_size,)
            
            volume_shape_correct = volumes.shape == expected_volume_shape
            labels_shape_correct = labels.shape == expected_labels_shape
            labels_valid = all(0 <= l < len(class_to_idx) for l in labels.numpy())
            
            return self.add_result(
                "Dummy Dataset",
                volume_shape_correct and labels_shape_correct and labels_valid,
                f"Successfully created dummy dataset and loaded batch",
                {
                    "batch_size": batch_size,
                    "volume_shape": list(volumes.shape),
                    "expected_volume_shape": list(expected_volume_shape),
                    "labels_shape": list(labels.shape),
                    "expected_labels_shape": list(expected_labels_shape),
                    "sample_labels": labels.tolist(),
                    "sample_participant_ids": participant_ids,
                    "volume_shape_correct": volume_shape_correct,
                    "labels_shape_correct": labels_shape_correct,
                    "labels_valid": labels_valid
                }
            )
            
        except Exception as e:
            return self.add_result(
                "Dummy Dataset",
                False,
                f"Dummy dataset test failed: {str(e)}",
                {"error": str(e)}
            )
    
    def check_encoder_loading(self, checkpoint_paths: List[str]) -> ValidationResult:
        """Test V-JEPA2 encoder loading from checkpoints."""
        try:
            loaded_encoders = []
            
            for checkpoint_path in checkpoint_paths:
                if not os.path.exists(checkpoint_path):
                    logger.warning(f"Checkpoint not found: {checkpoint_path}")
                    continue
                
                try:
                    # Load encoder
                    encoder = load_vjepa2_encoder(checkpoint_path, freeze=True)
                    
                    # Validate encoder
                    dummy_input = torch.randn(1, 1, 64, 384, 384)
                    validation_passed = validate_encoder_output(encoder, dummy_input.shape)
                    
                    checkpoint_name = os.path.basename(checkpoint_path)
                    loaded_encoders.append({
                        "checkpoint": checkpoint_name,
                        "path": checkpoint_path,
                        "validation_passed": validation_passed,
                        "embed_dim": encoder.embed_dim,
                        "num_patches": encoder.patch_embed.num_patches
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to load encoder from {checkpoint_path}: {e}")
            
            if not loaded_encoders:
                return self.add_result(
                    "Encoder Loading",
                    False,
                    "No encoders successfully loaded",
                    {"attempted_checkpoints": checkpoint_paths}
                )
            
            success_count = sum(1 for enc in loaded_encoders if enc["validation_passed"])
            
            return self.add_result(
                "Encoder Loading",
                success_count > 0,
                f"Loaded {success_count}/{len(loaded_encoders)} encoders successfully",
                {
                    "attempted_checkpoints": len(checkpoint_paths),
                    "loaded_encoders": len(loaded_encoders),
                    "validated_encoders": success_count,
                    "encoder_details": loaded_encoders
                }
            )
            
        except Exception as e:
            return self.add_result(
                "Encoder Loading",
                False,
                f"Encoder loading test failed: {str(e)}",
                {"error": str(e)}
            )
    
    def check_model_creation(self, checkpoint_path: str, config: Dict[str, Any]) -> ValidationResult:
        """Test complete model creation and forward pass."""
        try:
            if not os.path.exists(checkpoint_path):
                return self.add_result(
                    "Model Creation",
                    False,
                    f"Checkpoint not found: {checkpoint_path}",
                    {"path": checkpoint_path}
                )
            
            # Create model
            device = torch.device('cpu')  # Use CPU for testing
            model = create_model_from_checkpoint(
                checkpoint_path=checkpoint_path,
                config=config,
                freeze_encoder=True,
                device=device
            )
            
            # Test forward pass
            dummy_input = torch.randn(2, 1, 64, 384, 384)  # Batch of 2
            model.eval()
            
            with torch.no_grad():
                logits = model(dummy_input)
            
            # Validate output
            expected_shape = (2, config.get('classes', {}).get('mapping', {}) and len(config['classes']['mapping']) or 4)
            output_shape_correct = logits.shape == expected_shape
            output_finite = torch.isfinite(logits).all()
            
            # Get parameter counts
            param_counts = model.count_parameters()
            
            return self.add_result(
                "Model Creation",
                output_shape_correct and output_finite,
                f"Model created and forward pass successful",
                {
                    "checkpoint": os.path.basename(checkpoint_path),
                    "output_shape": list(logits.shape),
                    "expected_shape": list(expected_shape),
                    "output_shape_correct": output_shape_correct,
                    "output_finite": output_finite.item(),
                    "output_range": [float(logits.min()), float(logits.max())],
                    "parameter_counts": param_counts
                }
            )
            
        except Exception as e:
            return self.add_result(
                "Model Creation",
                False,
                f"Model creation test failed: {str(e)}",
                {"error": str(e), "checkpoint": checkpoint_path}
            )
    
    def run_smoke_tests(
        self,
        labels_path: str,
        checkpoint_paths: List[str],
        config: Dict[str, Any],
        test_b2: bool = True
    ) -> bool:
        """
        Run complete smoke test suite.
        
        Args:
            labels_path: Path to participants.tsv
            checkpoint_paths: List of V-JEPA2 checkpoint paths
            config: Model configuration
            test_b2: Whether to test B2 connection (requires credentials)
            
        Returns:
            True if all critical tests pass
        """
        logger.info("🧪 Starting smoke test suite...")
        
        # Test B2 connection (optional)
        if test_b2:
            self.check_b2_connection()
        
        # Test labels processing (critical)
        labels_result = self.check_labels_loading(labels_path)
        
        # Test OCT locator (critical if using real data)
        if test_b2:
            self.check_oct_locator()
        
        # Test dummy dataset creation (critical)
        dummy_dataset_result = self.check_dummy_dataset(labels_path)
        
        # Test encoder loading (critical)
        encoder_result = self.check_encoder_loading(checkpoint_paths)
        
        # Test model creation (critical)
        if checkpoint_paths and encoder_result.passed:
            model_result = self.check_model_creation(checkpoint_paths[0], config)
        
        # Summary
        passed_tests = sum(1 for r in self.results if r.passed)
        total_tests = len(self.results)
        
        logger.info(f"🧪 Smoke tests completed: {passed_tests}/{total_tests} passed")
        
        # Critical tests that must pass
        critical_results = [
            labels_result,
            dummy_dataset_result,
            encoder_result
        ]
        
        critical_passed = all(r.passed for r in critical_results)
        
        if critical_passed:
            logger.info("✅ All critical smoke tests passed - pipeline ready!")
        else:
            logger.error("❌ Critical smoke tests failed - pipeline needs fixes")
        
        return critical_passed
    
    def print_summary(self):
        """Print detailed test summary."""
        print("\n" + "="*60)
        print("SMOKE TEST SUMMARY")
        print("="*60)
        
        for result in self.results:
            print(result)
            if result.details:
                for key, value in result.details.items():
                    if isinstance(value, (dict, list)) and len(str(value)) > 100:
                        print(f"    {key}: {type(value).__name__}({len(value) if hasattr(value, '__len__') else '?'} items)")
                    else:
                        print(f"    {key}: {value}")
                print()
        
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        print(f"OVERALL: {passed}/{total} tests passed")
        print("="*60)


def run_pipeline_smoke_test(
    labels_path: str = "/Users/layne/Mac/Acdamic/UCInspire/3d_oct_fundation_model/fine-tuneing-data/participants.tsv",
    checkpoint_paths: List[str] = None,
    test_b2_connection: bool = False
) -> bool:
    """
    Run complete pipeline smoke test with default paths.
    
    Args:
        labels_path: Path to participants.tsv
        checkpoint_paths: List of checkpoint paths to test
        test_b2_connection: Whether to test B2 (requires credentials)
        
    Returns:
        True if all critical tests pass
    """
    # Default checkpoint paths
    if checkpoint_paths is None:
        checkpoint_base = "/Users/layne/Mac/Acdamic/UCInspire/checkpoints"
        checkpoint_paths = [
            f"{checkpoint_base}/best_checkpoint_multi_domain.pt",
            f"{checkpoint_base}/best_checkpoint_single_domain_01.pt",
            f"{checkpoint_base}/best_checkpoint_single_domain_02.pt"
        ]
    
    # Default config
    config = {
        'emb_dim': 768,
        'classes': {
            'mapping': {
                'healthy': 0,
                'pre_diabetes_lifestyle_controlled': 1,
                'oral_medication_and_or_non_insulin_injectable_medication_controlled': 2,
                'insulin_dependent': 3
            }
        },
        'head': {
            'hidden': 0,  # Linear probe
            'dropout': 0.1
        }
    }
    
    # Run tests
    suite = ValidationSuite()
    success = suite.run_smoke_tests(
        labels_path=labels_path,
        checkpoint_paths=checkpoint_paths,
        config=config,
        test_b2=test_b2_connection
    )
    
    suite.print_summary()
    return success


if __name__ == "__main__":
    # Run smoke tests
    import argparse
    
    parser = argparse.ArgumentParser(description="Run fine-tuning pipeline smoke tests")
    parser.add_argument("--labels", default="/Users/layne/Mac/Acdamic/UCInspire/3d_oct_fundation_model/fine-tuneing-data/participants.tsv")
    parser.add_argument("--test-b2", action="store_true", help="Test B2 connection (requires credentials)")
    
    args = parser.parse_args()
    
    success = run_pipeline_smoke_test(
        labels_path=args.labels,
        test_b2_connection=args.test_b2
    )
    
    sys.exit(0 if success else 1)