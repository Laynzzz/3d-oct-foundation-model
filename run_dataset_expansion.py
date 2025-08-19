#!/usr/bin/env python3
"""
One-time dataset expansion script for OCT foundation model.

This script expands ZIP files in GCS to individual DICOM files.
Based on section 3.7 of the plan.md.

Usage:
    # Dry run first (recommended)
    python run_dataset_expansion.py --dry-run
    
    # Full expansion
    python run_dataset_expansion.py
    
    # Validate after expansion
    python run_dataset_expansion.py --validate-only
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_setup.expand_gcs_dataset import main

if __name__ == "__main__":
    print("=== OCT Foundation Model - Dataset Expansion ===")
    print("This script will expand ZIP files in GCS to individual DICOM files.")
    print("See section 3.7 of plan.md for details.\n")
    
    main()