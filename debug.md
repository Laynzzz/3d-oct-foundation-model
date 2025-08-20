# DICOM Reading Issues - Debug Report

## Current Issue
Training is experiencing widespread DICOM file reading failures with error messages like:
```
WARNING:data_setup.datasets:Failed to load DICOM at index 2174: gs://layne-tpu-code-sync/OCTdata/OCTdata/retinal_oct/structural_oct/topcon_triton/7399/7399_triton_3d_radial_oct_r_2.16.840.1.114517.10.5.1.4.94005920240724155817.1.1.dcm
```

## Problem Analysis
- **Symptom**: Large number of DICOM files failing to load during training
- **Impact**: Significantly reduced effective dataset size, potential training instability
- **File Type**: OCT DICOM files with JPEG2000 compression
- **Location**: GCS bucket `gs://layne-tpu-code-sync/OCTdata/OCTdata/retinal_oct/structural_oct/`

## Potential Root Causes

### 1. JPEG Decompression Libraries
**Status**: ✅ VERIFIED INSTALLED
- `pylibjpeg==2.0.1` ✅
- `pylibjpeg-libjpeg==2.2.0` ✅  
- `pylibjpeg-openjpeg==2.3.0` ✅

### 2. DICOM Transfer Syntax Issues
**Likely Cause**: OCT files may use unsupported transfer syntaxes
- JPEG2000 Lossless/Lossy compression
- RLE compression
- Deflated Explicit VR Little Endian

### 3. File Corruption During GCS Transfer
**Possible**: Files may have been corrupted during ZIP extraction to individual DICOMs

### 4. Authentication/Permission Issues
**Less Likely**: Would affect all files, not just some

## Diagnostic Steps

### Step 1: Test Specific Failed File
```bash
# Upload debug script to TPU
gcloud compute tpus tpu-vm scp debug_dicom.py oct-jepa2-v4-32:~/3d-oct-foundation-model/ --zone=us-central2-b --worker=0

# Run diagnostic
gcloud compute tpus tpu-vm ssh oct-jepa2-v4-32 --zone=us-central2-b --worker=0 --command="export PATH=/home/layne/miniconda/envs/torch-xla/bin:\$PATH && cd ~/3d-oct-foundation-model && python debug_dicom.py"
```

### Step 2: Check Available DICOM Decoders
The debug script will check:
- Available pydicom pixel handlers
- JPEG library versions
- Specific error types for failed files

### Step 3: Sample Good vs Bad Files
Compare working vs failing files to identify patterns:
- Transfer syntax differences
- Compression types
- Metadata variations

## Potential Solutions

### Solution 1: Install Additional DICOM Libraries
```bash
# Install additional decompression support
gcloud compute tpus tpu-vm ssh oct-jepa2-v4-32 --zone=us-central2-b --worker=all --command="export PATH=/home/layne/miniconda/envs/torch-xla/bin:\$PATH && pip install pillow-simd gdcm-python"
```

### Solution 2: Force Decompression at Read Time
Modify `gcs_dicom_reader.py` to force pixel data decompression:
```python
# In read_dicom_volume method, before accessing pixel_array:
dataset.decompress()  # Force decompression
```

### Solution 3: Filter to Working Files Only
Create a validated file list excluding corrupted files:
```bash
# Create subset of verified working files
python create_validated_filelist.py --input manifest.tsv --output validated_manifest.tsv
```

### Solution 4: Fallback to Different Transfer Syntax
Add fallback logic to convert transfer syntax:
```python
# Convert to explicit VR little endian if compression fails
if hasattr(dataset, 'file_meta'):
    dataset.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
```

## Immediate Action Plan

1. **Upload and run debug script** to identify specific error types
2. **Check if it's a systematic issue** (all files failing) vs random corruption
3. **Try Solution 2** (force decompression) as it's least invasive
4. **If still failing, try Solution 1** (additional libraries)
5. **Last resort: Solution 3** (filter to working subset)

## Current Workaround
The training can continue with reduced dataset size due to robust error handling in `OCTDICOMDataset`. However, fixing this will:
- Increase effective training data
- Improve model performance  
- Ensure reproducible results

## Files to Modify
- `data_setup/gcs_dicom_reader.py` - Add force decompression
- `debug_dicom.py` - Diagnostic script (already created)
- Potentially `requirements.txt` - Additional libraries if needed

## Success Criteria
- DICOM read failure rate < 5%
- Training proceeds with full dataset
- No impact on training performance