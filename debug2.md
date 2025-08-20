Debugging Guide: No Valid Samples in Batch Error
📌 Error Summary
Training crashed with:
RuntimeError: No valid samples in batch - check participant range filtering
Torch/XLA then raised:
TypeError: 'mappingproxy' object does not support item assignment
Root cause: Empty batches caused by DICOM files missing pixel data.
🔎 What Happened
Data loader tried to read .dcm files.
Some files had no (7FE0,0010) Pixel Data element:
WARNING: No pixel data found...
All samples in the batch became None.
Collate function raised RuntimeError.
Torch/XLA crashed while handling the empty batch.
✅ Steps to Debug
Step 1 — Verify if DICOM files contain pixel data
import pydicom

path = "gs://layne-tpu-code-sync/OCTdata/OCTdata/retinal_oct/structural_oct/topcon_triton/1080/...dcm"

ds = pydicom.dcmread(path)
print("Has pixel data:", hasattr(ds, "PixelData"))
If False, the file is metadata-only or corrupted.
Step 2 — Batch-check all DICOMs
import os, pydicom

def check_dicom_folder(folder):
    bad_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".dcm"):
                try:
                    ds = pydicom.dcmread(os.path.join(root, f), stop_before_pixels=True)
                    if "PixelData" not in ds:
                        bad_files.append(f)
                except Exception as e:
                    bad_files.append(f"ERROR {f}: {e}")
    return bad_files

bad = check_dicom_folder("/local/path/to/OCTdata")
print("Bad DICOMs:", bad)
This will show which files lack pixel data.
Step 3 — Exclude invalid files
Update dataset loader to skip files without pixel data.
Example filter:
if "PixelData" not in ds:
    continue  # skip invalid sample
Step 4 — Check participant range filtering
Ensure filtering config isn’t discarding all valid samples.
Log how many samples remain after filtering.
⚠️ Notes
XLA_USE_BF16 deprecation warnings are harmless, but you should switch to bf16 config later.
Multiple W&B runs are retries; not part of the bug.
🎯 Conclusion
The main issue is invalid DICOM files (no pixel data).
Fix by detecting and skipping bad files before batching.
Once data loading works, Torch/XLA errors should disappear.

✅ FIXES APPLIED (August 20, 2025)

🔧 **Fix 1: Enhanced DICOM Validation in GCS Reader**
- Updated `gcs_dicom_reader.py` to check for PixelData tag (7FE0,0010) before accessing pixel_array
- Added robust error handling for corrupted pixel data
- Better logging for debugging invalid files

🔧 **Fix 2: Graceful Empty Batch Handling**  
- Updated `collate_fn()` in `datasets.py` to return None instead of raising RuntimeError
- Training loop already handles None batches with continue statement
- Prevents XLA multiprocessing crashes from empty batches

🔧 **Fix 3: Multiprocessing Compatibility** 
- Already fixed in training script with `multiprocessing.set_start_method('forkserver', force=True)`
- Resolves TypeError: 'mappingproxy' object issues with Python 3.11 + XLA 2.7.0

📋 **Testing Script Created**
- `test_data_pipeline_fix.py` - validates all fixes with sample data
- Tests single file loading and dataset/dataloader functionality
- Confirms pipeline works with invalid DICOM files present

🎯 **Next Steps**
Run the test script to validate fixes:
```bash
python test_data_pipeline_fix.py
```

If successful, proceed with full training using:
```bash
bash run_tpu_xla.sh configs/smoke_test.yaml
```