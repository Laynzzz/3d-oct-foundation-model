"""
Labels processing module for fine-tuning dataset.
Handles TSV parsing, class mapping, and train/val/test splits.
"""

import pandas as pd
from typing import Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)


def load_labels(tsv_path: str) -> pd.DataFrame:
    """
    Load labels from TSV file (local or B2).
    
    Args:
        tsv_path: Path to participants.tsv file
        
    Returns:
        DataFrame with participant metadata
    """
    try:
        # Handle B2 paths
        if tsv_path.startswith('ai-readi/') or tsv_path.startswith('eye-dataset/'):
            # B2 path - download the file first
            try:
                from ..storage.b2 import read_with_cache
                bucket_name = 'eye-dataset'
                key = tsv_path.replace('eye-dataset/', '') if tsv_path.startswith('eye-dataset/') else tsv_path
                
                # Download to temporary file
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(mode='w+b', suffix='.tsv', delete=False) as tmp_file:
                    data = read_with_cache(bucket_name, key)
                    tmp_file.write(data)
                    temp_path = tmp_file.name
                
                # Read from temporary file
                df = pd.read_csv(temp_path, sep='\t')
                
                # Clean up temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass  # Ignore cleanup errors
                    
                logger.info(f"Loaded {len(df)} participants from B2: {tsv_path}")
                return df
                
            except Exception as e:
                logger.error(f"Failed to load labels from B2 path {tsv_path}: {e}")
                raise
        else:
            # Local file path
            df = pd.read_csv(tsv_path, sep='\t')
            logger.info(f"Loaded {len(df)} participants from {tsv_path}")
            return df
    except Exception as e:
        logger.error(f"Failed to load labels from {tsv_path}: {e}")
        raise


def filter_oct_available(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter DataFrame to only include participants with OCT data available.
    
    Args:
        df: DataFrame with participant metadata
        
    Returns:
        Filtered DataFrame with only retinal_oct == True
    """
    if 'retinal_oct' not in df.columns:
        logger.warning("Column 'retinal_oct' not found in DataFrame")
        return df
    
    filtered_df = df[df['retinal_oct'] == True].copy()
    logger.info(f"Filtered to {len(filtered_df)} participants with OCT data")
    return filtered_df


def map_classes(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Map study groups to numeric class labels.
    
    Args:
        df: DataFrame with 'study_group' column
        
    Returns:
        Tuple of (updated DataFrame with 'class_label' column, class_to_idx mapping)
    """
    class_to_idx = {
        'healthy': 0,
        'pre_diabetes_lifestyle_controlled': 1,
        'oral_medication_and_or_non_insulin_injectable_medication_controlled': 2,
        'insulin_dependent': 3
    }
    
    if 'study_group' not in df.columns:
        raise ValueError("Column 'study_group' not found in DataFrame")
    
    df = df.copy()
    df['class_label'] = df['study_group'].map(class_to_idx)
    
    # Check for unmapped classes
    unmapped = df['class_label'].isna()
    if unmapped.any():
        unmapped_groups = df[unmapped]['study_group'].unique()
        logger.warning(f"Unmapped study groups found: {unmapped_groups}")
        # Drop unmapped rows
        df = df.dropna(subset=['class_label'])
        df['class_label'] = df['class_label'].astype(int)
    
    # Log class distribution
    class_counts = df['class_label'].value_counts().sort_index()
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    logger.info("Class distribution:")
    for class_idx, count in class_counts.items():
        class_name = idx_to_class[class_idx]
        logger.info(f"  {class_idx} ({class_name}): {count} samples")
    
    return df, class_to_idx


def split_by_recommendation(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame by recommended_split column.
    
    Args:
        df: DataFrame with 'recommended_split' column
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if 'recommended_split' not in df.columns:
        raise ValueError("Column 'recommended_split' not found in DataFrame")
    
    train_df = df[df['recommended_split'] == 'train'].copy()
    val_df = df[df['recommended_split'] == 'val'].copy()
    test_df = df[df['recommended_split'] == 'test'].copy()
    
    logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Log class distribution per split
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        if len(split_df) > 0 and 'class_label' in split_df.columns:
            class_counts = split_df['class_label'].value_counts().sort_index()
            logger.info(f"{split_name} class distribution: {dict(class_counts)}")
    
    return train_df, val_df, test_df


def get_class_weights(df: pd.DataFrame, method: str = 'balanced') -> Dict[int, float]:
    """
    Calculate class weights for handling class imbalance.
    
    Args:
        df: DataFrame with 'class_label' column
        method: 'balanced' for sklearn-style balanced weights, 'uniform' for equal weights
        
    Returns:
        Dictionary mapping class indices to weights
    """
    if 'class_label' not in df.columns:
        raise ValueError("Column 'class_label' not found in DataFrame")
    
    class_counts = df['class_label'].value_counts().sort_index()
    num_classes = len(class_counts)
    
    if method == 'balanced':
        # Sklearn-style balanced class weights: n_samples / (n_classes * np.bincount(y))
        total_samples = len(df)
        weights = {}
        for class_idx, count in class_counts.items():
            weights[class_idx] = total_samples / (num_classes * count)
    elif method == 'uniform':
        # All classes weighted equally
        weights = {class_idx: 1.0 for class_idx in class_counts.index}
    else:
        raise ValueError(f"Unknown method: {method}")
    
    logger.info(f"Class weights ({method}): {weights}")
    return weights


def process_labels(tsv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[int, float]]:
    """
    Complete labels processing pipeline.
    
    Args:
        tsv_path: Path to participants.tsv file
        
    Returns:
        Tuple of (train_df, val_df, test_df, class_to_idx, class_weights)
    """
    logger.info("Starting labels processing pipeline")
    
    # Load and filter data
    df = load_labels(tsv_path)
    df = filter_oct_available(df)
    
    # Map classes
    df, class_to_idx = map_classes(df)
    
    # Split data
    train_df, val_df, test_df = split_by_recommendation(df)
    
    # Calculate class weights from training set
    class_weights = get_class_weights(train_df, method='balanced')
    
    logger.info("Labels processing completed")
    return train_df, val_df, test_df, class_to_idx, class_weights


def validate_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
    """
    Validate that splits don't have overlapping participants.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame  
        test_df: Test DataFrame
        
    Returns:
        True if splits are valid, False otherwise
    """
    if 'participant_id' not in train_df.columns:
        logger.warning("Cannot validate splits: 'participant_id' column not found")
        return True
    
    train_ids = set(train_df['participant_id'])
    val_ids = set(val_df['participant_id']) 
    test_ids = set(test_df['participant_id'])
    
    # Check for overlaps
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids
    
    if train_val_overlap:
        logger.error(f"Train/Val overlap: {train_val_overlap}")
        return False
    
    if train_test_overlap:
        logger.error(f"Train/Test overlap: {train_test_overlap}")
        return False
        
    if val_test_overlap:
        logger.error(f"Val/Test overlap: {val_test_overlap}")
        return False
    
    logger.info("Split validation passed - no overlapping participants")
    return True