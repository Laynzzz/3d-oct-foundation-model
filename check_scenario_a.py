#!/usr/bin/env python3
"""
Quick check for Scenario A: Single class prediction
Run this while training continues to diagnose validation issue
"""

import sys
import os
sys.path.append('/Users/layne/Mac/Acdamic/UCInspire/3d_oct_fundation_model')

from finetuning.data.labels import load_labels_and_splits
from collections import Counter

def check_scenario_a():
    """
    Quick mathematical check without loading models
    """
    print("🔍 SCENARIO A CHECK: Single Class Prediction")
    print("=" * 55)
    
    # Load the actual validation split
    try:
        s3_config = {
            'endpoint_env': 'S3_ENDPOINT_URL',
            'bucket': 'eye-dataset',
            'labels_tsv': 'ai-readi/dataset/participants.tsv'
        }
        
        _, val_labels, _ = load_labels_and_splits(s3_config)
        
        print(f"\n📊 VALIDATION SET ANALYSIS:")
        print(f"Total validation samples: {len(val_labels)}")
        
        # Count classes in validation
        val_class_counts = Counter(val_labels)
        print(f"Val class distribution: {dict(val_class_counts)}")
        
        # Calculate what accuracy would be if predicting each class
        print(f"\n🎯 ACCURACY IF ALWAYS PREDICTING:")
        for class_id, count in val_class_counts.items():
            accuracy = count / len(val_labels)
            print(f"Class {class_id}: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Check if 26.2% matches any single class
        target_accuracy = 0.262
        print(f"\n🔍 OBSERVED VALIDATION ACCURACY: {target_accuracy:.3f}")
        
        for class_id, count in val_class_counts.items():
            class_accuracy = count / len(val_labels)
            if abs(class_accuracy - target_accuracy) < 0.01:  # Within 1%
                print(f"✅ MATCH: Always predicting class {class_id}")
                print(f"   Expected: {class_accuracy:.3f}, Observed: {target_accuracy:.3f}")
                return class_id
        
        print("❓ No exact match - might be mixed prediction or other issue")
        return None
        
    except Exception as e:
        print(f"❌ Error loading validation data: {e}")
        
        # Fallback: Use the logged class distribution
        print(f"\n📋 FALLBACK ANALYSIS (from logs):")
        print("Val class distribution from logs: {0: 39, 1: 40, 2: 47, 3: 33}")
        
        total = 159  # From logs
        for class_id, count in [(0,39), (1,40), (2,47), (3,33)]:
            accuracy = count / total
            print(f"Class {class_id}: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Check 26.2%
        print(f"\n🔍 CHECKING 26.2% ACCURACY:")
        for class_id, count in [(0,39), (1,40), (2,47), (3,33)]:
            accuracy = count / total
            if abs(accuracy - 0.262) < 0.01:
                print(f"✅ MATCH: Always predicting class {class_id}")
                return class_id
        
        print("❓ No exact match with 26.2%")
        return None

if __name__ == "__main__":
    predicted_class = check_scenario_a()
    
    if predicted_class is not None:
        print(f"\n🚨 DIAGNOSIS CONFIRMED: Scenario A")
        print(f"Model is likely always predicting class {predicted_class}")
        print(f"\n💡 IMMEDIATE FIXES NEEDED:")
        print(f"1. Add class weights to loss function")
        print(f"2. Increase head capacity (hidden: 256)")  
        print(f"3. Increase head learning rate (lr_head: 0.003)")
        print(f"4. Add gradient accumulation for effective batch size 8-16")
    else:
        print(f"\n🤔 SCENARIO A NOT CONFIRMED")
        print(f"Need deeper investigation - run full confusion matrix")