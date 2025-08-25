#!/usr/bin/env python3
"""
Debug validation accuracy - check if model is predicting single class
Based on finetune-fix.md line 495+ analysis
"""

import torch
import numpy as np
from collections import Counter
import torch.nn.functional as F

def analyze_validation_predictions(model, val_loader, device='cpu'):
    """
    Quick triage to check if model is predicting single class
    """
    print("🔍 VALIDATION PREDICTION ANALYSIS")
    print("=" * 50)
    
    model.eval()
    all_y, all_p, all_logits = [], [], []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(val_loader):
            if batch_idx >= 10:  # Quick sample for diagnosis
                break
            x, y = x.to(device), y.to(device)
            logits = model(x)
            
            all_p.append(logits.argmax(1).cpu())
            all_y.append(y.cpu())
            all_logits.append(logits.cpu())
    
    y = torch.cat(all_y).numpy()
    p = torch.cat(all_p).numpy()
    logits = torch.cat(all_logits)
    
    # 1) Class frequency & "all-one-class" check
    print("\n📊 CLASS DISTRIBUTION:")
    print("Validation true class counts:", Counter(y))
    print("Predicted class counts:", Counter(p))
    
    # 2) Confusion matrix
    print("\n🔄 CONFUSION MATRIX:")
    C = np.zeros((4,4), dtype=int)
    for yi, pi in zip(y, p): 
        C[yi, pi] += 1
    print("True\\Pred  0    1    2    3")
    for i in range(4):
        print(f"Class {i}   {C[i,0]:3d}  {C[i,1]:3d}  {C[i,2]:3d}  {C[i,3]:3d}")
    
    # 3) Per-class accuracy
    print("\n🎯 PER-CLASS ACCURACY:")
    per_class = C.diagonal() / np.maximum(C.sum(axis=1), 1)
    for i, acc in enumerate(per_class):
        print(f"Class {i}: {acc:.3f}")
    
    # 4) Prediction entropy (collapse check)
    print("\n🌀 PREDICTION ENTROPY:")
    probs = F.softmax(logits, 1)
    ent = -(probs * torch.log(probs + 1e-8)).sum(1)
    print(f"Mean prediction entropy: {ent.mean():.3f} (max=1.386 for uniform)")
    
    # 5) Confidence analysis
    print("\n📈 CONFIDENCE ANALYSIS:")
    max_probs = probs.max(1)[0]
    print(f"Mean confidence: {max_probs.mean():.3f}")
    print(f"Min confidence: {max_probs.min():.3f}")
    print(f"Max confidence: {max_probs.max():.3f}")
    
    # 6) Diagnosis
    print("\n🔬 DIAGNOSIS:")
    pred_classes = set(p)
    if len(pred_classes) == 1:
        print(f"❌ PROBLEM: Model only predicts class {pred_classes.pop()}")
        print("   Solution: Add class weights, increase head capacity")
    elif len(pred_classes) == 2:
        print(f"⚠️  ISSUE: Model only predicts 2 classes: {pred_classes}")
        print("   Solution: Check class weights and head learning rate")
    elif ent.mean() < 0.5:
        print("⚠️  ISSUE: Very low prediction entropy - model overconfident")
        print("   Solution: Add label smoothing, reduce overfitting")
    else:
        print("✅ Predictions look reasonable - check other factors")
    
    return C, per_class, ent.mean().item()

if __name__ == "__main__":
    print("Run this in the training environment with loaded model and val_loader")
    print("Example:")
    print("analyze_validation_predictions(model, val_loader, device)")