#!/usr/bin/env python3
"""
Comprehensive dataset scaling script that implements all three approaches:
1. Hybrid manufacturer filtering
2. Enhanced DICOM validation  
3. Systematic scaling to 500-1000 files
"""

import pandas as pd
import numpy as np
import subprocess
import time
import logging
from pathlib import Path
from data_setup.enhanced_dicom_validator import validate_file_batch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatasetScaler:
    """Comprehensive dataset scaling with multiple strategies."""
    
    def __init__(self, gcs_root: str, target_size: int = 500):
        self.gcs_root = gcs_root
        self.target_size = target_size
        self.validation_batch_size = 50
        
    def load_original_manifest(self) -> pd.DataFrame:
        """Load the original manifest."""
        manifest_path = "/tmp/original_manifest.tsv"
        df = pd.read_csv(manifest_path, sep='\t')
        logger.info(f"Loaded original manifest with {len(df)} files")
        return df
    
    def strategy_1_manufacturer_filtering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strategy 1: Extract files by manufacturer reliability."""
        
        logger.info("🎯 Strategy 1: Manufacturer-based filtering")
        
        # Tier 1: Heidelberg (100% success rate observed)
        heidelberg_df = df[df['manufacturer'].str.contains('Heidelberg', case=False, na=False)]
        logger.info(f"Heidelberg files: {len(heidelberg_df)}")
        
        # Tier 2: Other reliable manufacturers
        topcon_df = df[df['manufacturer'].str.contains('Topcon', case=False, na=False)]
        zeiss_df = df[df['manufacturer'].str.contains('Zeiss', case=False, na=False)]
        logger.info(f"Topcon files: {len(topcon_df)}")
        logger.info(f"Zeiss files: {len(zeiss_df)}")
        
        # Create stratified samples
        reliable_files = []
        
        # Take all Heidelberg (highest reliability)
        if len(heidelberg_df) > 0:
            reliable_files.append(heidelberg_df)
            logger.info(f"Added all {len(heidelberg_df)} Heidelberg files")
        
        # Sample from Topcon and Zeiss proportionally
        remaining_target = max(0, self.target_size - len(heidelberg_df))
        
        if remaining_target > 0 and len(topcon_df) > 0:
            topcon_sample_size = min(len(topcon_df), remaining_target // 2)
            topcon_sample = topcon_df.sample(n=topcon_sample_size, random_state=42)
            reliable_files.append(topcon_sample)
            logger.info(f"Sampled {len(topcon_sample)} Topcon files")
            remaining_target -= len(topcon_sample)
        
        if remaining_target > 0 and len(zeiss_df) > 0:
            zeiss_sample_size = min(len(zeiss_df), remaining_target)
            zeiss_sample = zeiss_df.sample(n=zeiss_sample_size, random_state=42)
            reliable_files.append(zeiss_sample)
            logger.info(f"Sampled {len(zeiss_sample)} Zeiss files")
        
        if reliable_files:
            result_df = pd.concat(reliable_files, ignore_index=True)
            logger.info(f"Strategy 1 result: {len(result_df)} files")
            return result_df
        else:
            logger.warning("No reliable manufacturer files found!")
            return pd.DataFrame()
    
    def strategy_2_enhanced_validation(self, df: pd.DataFrame, max_files: int = 200) -> pd.DataFrame:
        """Strategy 2: Enhanced validation on random sample."""
        
        logger.info(f"🔧 Strategy 2: Enhanced validation on {max_files} files")
        
        # Random sample for validation
        sample_df = df.sample(n=min(len(df), max_files), random_state=42)
        
        # Build file paths
        file_paths = [f"{self.gcs_root}{row['filepath']}" for _, row in sample_df.iterrows()]
        
        # Validate with enhanced validator
        logger.info("Running enhanced validation...")
        validation_results = validate_file_batch(file_paths, max_workers=16)
        
        valid_files = validation_results['valid_files']
        success_rate = validation_results['success_rate']
        
        logger.info(f"Enhanced validation results:")
        logger.info(f"  Success rate: {success_rate:.1%}")
        logger.info(f"  Valid files: {len(valid_files)}")
        
        # Filter manifest to keep only valid files
        valid_filepaths = [path.replace(self.gcs_root, '') for path in valid_files]
        valid_df = sample_df[sample_df['filepath'].isin(valid_filepaths)]
        
        logger.info(f"Strategy 2 result: {len(valid_df)} files")
        return valid_df
    
    def strategy_3_systematic_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strategy 3: Systematic scaling with progressive validation."""
        
        logger.info(f"📈 Strategy 3: Systematic scaling to {self.target_size} files")
        
        validated_files = []
        batch_size = self.validation_batch_size
        total_tested = 0
        
        # Shuffle the dataframe for random sampling
        df_shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        
        while len(validated_files) < self.target_size and total_tested < len(df_shuffled):
            # Get next batch
            start_idx = total_tested
            end_idx = min(start_idx + batch_size, len(df_shuffled))
            batch_df = df_shuffled.iloc[start_idx:end_idx]
            
            logger.info(f"Testing batch {total_tested//batch_size + 1}: files {start_idx+1}-{end_idx}")
            
            # Build file paths for this batch
            file_paths = [f"{self.gcs_root}{row['filepath']}" for _, row in batch_df.iterrows()]
            
            # Validate batch
            validation_results = validate_file_batch(file_paths, max_workers=8)
            valid_files = validation_results['valid_files']
            
            # Add valid files to our collection
            if valid_files:
                valid_filepaths = [path.replace(self.gcs_root, '') for path in valid_files]
                batch_valid_df = batch_df[batch_df['filepath'].isin(valid_filepaths)]
                validated_files.append(batch_valid_df)
                
                current_total = sum(len(df) for df in validated_files)
                logger.info(f"  ✅ Found {len(valid_files)} valid files (total: {current_total})")
                
                if current_total >= self.target_size:
                    logger.info(f"🎯 Reached target of {self.target_size} files!")
                    break
            else:
                logger.info(f"  ❌ No valid files in this batch")
            
            total_tested = end_idx
        
        if validated_files:
            result_df = pd.concat(validated_files, ignore_index=True)
            # Trim to exact target size
            if len(result_df) > self.target_size:
                result_df = result_df.head(self.target_size)
            
            logger.info(f"Strategy 3 result: {len(result_df)} files")
            return result_df
        else:
            logger.warning("No valid files found in systematic scaling!")
            return pd.DataFrame()
    
    def combine_strategies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine all strategies for maximum coverage."""
        
        logger.info("🔄 Combining all strategies")
        
        all_results = []
        
        # Strategy 1: Manufacturer filtering
        try:
            strategy1_df = self.strategy_1_manufacturer_filtering(df)
            if len(strategy1_df) > 0:
                all_results.append(strategy1_df)
        except Exception as e:
            logger.error(f"Strategy 1 failed: {e}")
        
        # Strategy 2: Enhanced validation (on remaining manufacturers)
        try:
            # Get files not in strategy 1
            if all_results:
                used_files = set(all_results[0]['filepath'])
                remaining_df = df[~df['filepath'].isin(used_files)]
            else:
                remaining_df = df
            
            if len(remaining_df) > 0:
                strategy2_df = self.strategy_2_enhanced_validation(remaining_df, max_files=100)
                if len(strategy2_df) > 0:
                    all_results.append(strategy2_df)
        except Exception as e:
            logger.error(f"Strategy 2 failed: {e}")
        
        # Strategy 3: Fill remaining with systematic scaling
        try:
            current_total = sum(len(df) for df in all_results)
            remaining_needed = max(0, self.target_size - current_total)
            
            if remaining_needed > 0:
                # Get files not yet included
                if all_results:
                    used_files = set()
                    for result_df in all_results:
                        used_files.update(result_df['filepath'])
                    remaining_df = df[~df['filepath'].isin(used_files)]
                else:
                    remaining_df = df
                
                if len(remaining_df) > 0:
                    # Temporarily adjust target for strategy 3
                    original_target = self.target_size
                    self.target_size = remaining_needed
                    
                    strategy3_df = self.strategy_3_systematic_scaling(remaining_df)
                    if len(strategy3_df) > 0:
                        all_results.append(strategy3_df)
                    
                    # Restore original target
                    self.target_size = original_target
        except Exception as e:
            logger.error(f"Strategy 3 failed: {e}")
        
        # Combine all results
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            # Remove duplicates (should be rare but possible)
            combined_df = combined_df.drop_duplicates(subset=['filepath'], keep='first')
            
            logger.info(f"🎯 Combined result: {len(combined_df)} files")
            return combined_df
        else:
            logger.error("All strategies failed!")
            return pd.DataFrame()
    
    def save_scaled_dataset(self, df: pd.DataFrame, suffix: str = "scaled"):
        """Save the scaled dataset to GCS."""
        
        if len(df) == 0:
            logger.error("Cannot save empty dataset!")
            return None
        
        # Save locally first
        output_path = f"/tmp/manifest_{suffix}.tsv"
        df.to_csv(output_path, sep='\t', index=False)
        
        # Upload to GCS
        gcs_output = f"gs://layne-tpu-code-sync/OCTdata/OCTdata/manifest_{suffix}.tsv"
        subprocess.run(['gsutil', 'cp', output_path, gcs_output], check=True)
        
        logger.info(f"✅ Saved scaled dataset:")
        logger.info(f"   Files: {len(df)}")
        logger.info(f"   Path: {gcs_output}")
        
        # Show breakdown
        logger.info(f"   Manufacturer breakdown:")
        breakdown = df.groupby('manufacturer').size()
        for mfg, count in breakdown.items():
            logger.info(f"     {mfg}: {count} files")
        
        return gcs_output

def main():
    """Main function to scale up the dataset."""
    
    logger.info("🚀 Starting Comprehensive Dataset Scaling")
    logger.info("=" * 60)
    
    # Configuration
    gcs_root = "gs://layne-tpu-code-sync/OCTdata/OCTdata"
    target_sizes = [100, 500, 1000]  # Multiple target sizes
    
    # Load original manifest
    scaler = DatasetScaler(gcs_root, target_size=500)
    df = scaler.load_original_manifest()
    
    # Create datasets of different sizes
    for target_size in target_sizes:
        logger.info(f"\n📊 Creating dataset with target size: {target_size}")
        logger.info("-" * 40)
        
        scaler.target_size = target_size
        start_time = time.time()
        
        scaled_df = scaler.combine_strategies(df)
        
        if len(scaled_df) > 0:
            suffix = f"scaled_{target_size}"
            gcs_path = scaler.save_scaled_dataset(scaled_df, suffix)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Created {suffix} dataset in {elapsed:.1f}s")
            logger.info(f"   Actual size: {len(scaled_df)} files")
            logger.info(f"   Target size: {target_size} files")
            logger.info(f"   Success rate: {len(scaled_df)/target_size:.1%}")
        else:
            logger.error(f"❌ Failed to create dataset with target size {target_size}")
    
    logger.info(f"\n🎉 Dataset scaling complete!")
    logger.info(f"   Available datasets:")
    logger.info(f"     manifest_minimal.tsv (20 files)")
    logger.info(f"     manifest_scaled_100.tsv (~100 files)")
    logger.info(f"     manifest_scaled_500.tsv (~500 files)")
    logger.info(f"     manifest_scaled_1000.tsv (~1000 files)")

if __name__ == "__main__":
    main()