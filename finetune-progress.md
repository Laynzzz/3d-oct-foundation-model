# Fine-Tuning Progress Report

**Date**: August 27, 2025  
**Task**: V-JEPA2 Single-Domain Fine-Tuning for OCT Diabetes Classification  
**Status**: 🚨 **CRITICAL DATA ISSUE IDENTIFIED**

## 🎯 **Objective**
Fine-tune single-domain V-JEPA2 checkpoint with Scenario A fixes to resolve single-class collapse issue (validation accuracy stuck at 26.15%).

## ✅ **Completed Achievements**

### 1. **XLA Distributed Training Implementation** ✅ **COMPLETE**
- **Problem**: Fine-tuning code lacked proper XLA multiprocessing like pretraining
- **Solution**: 
  - Refactored `finetuning/train/run.py` to use argparse pattern like pretraining
  - Created `run_finetune_xla.sh` shell script matching pretraining approach
  - Fixed multiprocessing to run on `--worker=all` with proper coordination
- **Result**: XLA distributed training working across 16 TPU cores

### 2. **Scenario A Fixes Implementation** ✅ **COMPLETE**
All critical fixes from `fine-tuning-single-fix.md` implemented:
- ✅ **Class Weighting**: Balanced weights computed from training data
- ✅ **MLP Head**: 768→256→4 with GELU activation (vs linear probe)
- ✅ **Progressive Unfreezing**: Encoder unfreezes at epoch 3
- ✅ **Label Smoothing**: 0.05 smoothing factor
- ✅ **Balanced Metrics**: Early stopping on balanced accuracy
- ✅ **Gradient Accumulation**: Effective batch size 16
- ✅ **Higher Learning Rates**: lr_head=3e-3, lr_encoder=1e-5

### 3. **Training Infrastructure** ✅ **WORKING**
- **Checkpointing**: Model checkpoints being saved (372MB checkpoint created)
- **W&B Monitoring**: Multiple distributed runs tracking metrics
- **B2 Data Integration**: Successfully connecting to eye-dataset bucket
- **Environment**: All B2 credentials and TPU environment properly configured

## 🚨 **CRITICAL ISSUE DISCOVERED**

### **Problem: Massive Data Loss Due to ID Mismatch**

**Symptoms:**
- Train dataset: 743 → 255 samples (66% loss)
- Validation dataset: 159 → 65 samples (59% loss) 
- W&B validation metrics all zero
- sklearn warning: "Only one class is present in y_true"

**Root Cause Analysis:**

#### **Expected vs Actual Data Structure:**
**Expected B2 Structure:**
```
eye-dataset/ai-readi/dataset/retinal_oct/structural_oct/
├── heidelberg_spectralis/1001/, 1002/, ... (1000 participants)
├── topcon_maestro2/1001/, 1002/, ... (1000 participants)  
├── topcon_triton/1001/, 1002/, ... (1000 participants)
└── zeiss_cirrus/1001/, 1002/, ... (1000 participants)
Total: 4000 participants with OCT data
```

**Labels File:**
- `participants.tsv`: 1061 participants with diabetes labels
- Participant IDs: 4-digit numbers (1001-1xxx range)

**The Mismatch:**
- **B2 Discovery**: Found 4000 participants across 4 devices ✅
- **Labels Available**: 1061 participants ✅  
- **Successfully Matched**: Only 255-320 participants ❌
- **Match Rate**: ~6-25% - **MASSIVE DATA LOSS**

#### **Impact on Training:**
1. **Severe Dataset Reduction**: Only ~25% of labeled participants have OCT data
2. **Class Imbalance**: Filtered validation set may contain only 1 class
3. **Metrics Failure**: ROC AUC, balanced accuracy fail with single class
4. **Non-Representative Sample**: Training on tiny subset, not full cohort

## 📊 **Training Attempt Results**

### **Technical Success:**
- ✅ XLA distributed training running across 4 TPU workers
- ✅ Data pipeline processing 255 train / 65 validation samples
- ✅ Model training and checkpoint saving working
- ✅ All Scenario A fixes active (class weighting, MLP head, etc.)

### **Critical Data Failure:**
- ❌ Validation metrics all zero (W&B dashboard)
- ❌ Only one class present in validation batches
- ❌ 66% of training data lost to ID mismatches
- ❌ Cannot validate if Scenario A fixes are working

## 🔍 **Technical Investigation**

### **Data Locator Analysis:**
The `finetuning/data/locator.py` discovers OCT data using:
- **Pattern Matching**: `\b(1[0-9]{3})\b` (4-digit IDs starting with 1)
- **File Formats**: `.dcm`, `.dicom`, `.nii`, `.nii.gz`, `.npy`, `.npz`
- **Directory Structure**: Hierarchical device/participant/files

### **ID Extraction Logic:**
```python
# Looks for participant IDs in paths like:
# ai-readi/dataset/retinal_oct/structural_oct/heidelberg_spectralis/1001/file.dcm
participant_id = re.search(r'\b(1[0-9]{3})\b', key).group(1)
```

### **Data Filtering Process:**
```python
def _filter_available_data(self):
    available_participants = set(self.locator.get_available_participants())  # ~4000 IDs
    self.df = self.df[self.df['participant_id'].isin(available_participants)]  # Only ~255 match
```

## 🎯 **Next Steps Required**

### **Priority 1: Data Investigation** 🚨 **URGENT**
1. **Inspect Participant ID Formats**:
   - Sample participant IDs from `participants.tsv`
   - Sample participant IDs extracted from B2 file paths
   - Identify format differences (leading zeros, prefixes, etc.)

2. **Analyze Data Availability**:
   - Which participants have labels but no OCT data?
   - Which participants have OCT data but no labels?
   - What's the overlap rate?

3. **Fix ID Matching Logic**:
   - Update participant ID extraction if format is different
   - Handle ID normalization (e.g., 1001 vs 001001)
   - Verify correct mapping between labels and files

### **Priority 2: Validation** 
Once data issue is resolved:
1. **Rerun Training**: With full dataset (should be 743/159 samples)
2. **Monitor Metrics**: Confirm validation metrics are non-zero
3. **Validate Fixes**: Confirm Scenario A fixes resolve single-class collapse
4. **Compare Results**: Target >30% validation accuracy vs 26.15% baseline

## 📈 **Expected Outcomes After Fix**

### **Dataset Size Recovery:**
- **Training**: 743 samples (vs current 255) - 3x increase
- **Validation**: 159 samples (vs current 65) - 2.5x increase
- **Class Distribution**: Balanced across all 4 diabetes classes

### **Metrics Recovery:**
- **Validation Metrics**: Non-zero ROC AUC, balanced accuracy, per-class F1
- **W&B Dashboard**: Proper metric tracking and curves
- **Scenario A Resolution**: Multi-class predictions instead of single-class collapse

### **Performance Target:**
- **Current Baseline**: 26.15% validation accuracy (single-class prediction)
- **Target with Fixes**: 35-50% balanced accuracy with all classes > 0.2 F1

## 🛠 **Implementation Status**

### **Core Infrastructure** ✅ **COMPLETE**
- [x] XLA distributed training
- [x] Scenario A fixes (class weighting, MLP head, progressive unfreezing)
- [x] B2 data integration
- [x] W&B monitoring
- [x] Checkpointing and early stopping

### **Critical Data Issue** ⏳ **IN PROGRESS** 
- [ ] Investigate participant ID mismatch
- [ ] Fix data locator ID extraction
- [ ] Validate full dataset availability
- [ ] Rerun training with complete data

---

**Status**: Training infrastructure working perfectly, but critical data availability issue preventing proper evaluation of Scenario A fixes. 

**Next Action**: Investigate and resolve participant ID mismatch to unlock full dataset for training.

**Owner**: Tianyu Xia  
**Last Updated**: August 27, 2025 5:05 AM EST