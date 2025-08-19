#!/usr/bin/env python3
"""
Quick inline dataset expansion script - exactly as shown in plan.md section 3.7.

This is the minimal version for direct execution on TPU VM.
"""

import gcsfs
import zipfile
import io

def main():
    print("Starting dataset expansion...")
    
    fs = gcsfs.GCSFileSystem()
    SRC = 'gs://layne-tpu-code-sync/OCTdata/OCTdata'
    DST = SRC + '/retinal_oct/structural_oct'
    
    for dev in ['heidelberg_spectralis','topcon_triton','topcon_maestro2','zeiss_cirrus']:
        print(f"Processing device: {dev}")
        
        for zpath in fs.glob(f"{SRC}/{dev}/*.zip"):
            print(f"  Extracting: {zpath}")
            
            with fs.open(zpath, 'rb') as f:
                data = f.read()  # for very large zips, prefer Option A
            
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                dcm_count = 0
                for name in zf.namelist():
                    if not name.lower().endswith('.dcm'): 
                        continue
                    
                    dst = f"{DST}/{dev}/" + name.lstrip('./')
                    fs.makedirs(dst.rsplit('/',1)[0], exist_ok=True)
                    
                    with zf.open(name) as zfh, fs.open(dst, 'wb') as out:
                        out.write(zfh.read())
                    
                    dcm_count += 1
                    if dcm_count % 50 == 0:
                        print(f"    Extracted {dcm_count} DICOM files...")
                
                print(f"  Completed: {dcm_count} DICOM files extracted")
    
    print('Done')

if __name__ == "__main__":
    main()