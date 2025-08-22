#!/usr/bin/env python3
"""
Create baseline snapshot - Step 1 of stabilization plan
Captures current stable state at step 1113+ without interrupting training
"""

import time
import json
from pathlib import Path

def create_baseline_snapshot():
    """Create baseline snapshot of current stable training state"""
    
    # Create snapshot directory
    timestamp = int(time.time())
    snapshot_dir = Path(f"baseline_snapshot_{timestamp}")
    snapshot_dir.mkdir(exist_ok=True)
    
    print(f"Creating baseline snapshot in: {snapshot_dir}")
    
    # Record current state
    snapshot_info = {
        "timestamp": timestamp,
        "step_range": "1113+",
        "status": "stable - passed critical step 514 without NaN",
        "a_prime_fix": "applied and working",
        "config": {
            "lr": "constant 1e-4 (base_lr)",
            "grad_clip": 0.01,
            "workers": 0,
            "persistent_workers": False,
            "pin_memory": False,
            "torch_compile_disabled": True
        },
        "key_metrics_to_verify": [
            "train/lr should be flat at 1e-4",
            "train/grad_norm should be finite and >0", 
            "train/loss should be trending/stable",
            "no train/grad_nonfinite events",
            "no N/A or NaN losses"
        ],
        "next_steps": [
            "1. Wait for next checkpoint save (every 5 epochs or at end)",
            "2. Verify current metrics are XLA-friendly", 
            "3. Add monotonic cosine scheduler gradually",
            "4. Optimize throughput one change at a time"
        ]
    }
    
    # Save snapshot info
    with open(snapshot_dir / "snapshot_info.json", "w") as f:
        json.dump(snapshot_info, f, indent=2)
    
    # Create instructions for manual checkpoint capture
    instructions = f"""
BASELINE SNAPSHOT INSTRUCTIONS
=============================
Timestamp: {timestamp}
Directory: {snapshot_dir}

CURRENT STATUS:
- Training running at step 1113+ 
- A-prime fix successful (no NaN at step 514)
- Ready to capture baseline

NEXT ACTIONS (do these when convenient):

1. WAIT for next automatic checkpoint save
   - Training saves every 5 epochs or at completion
   - Check /tmp/checkpoints/ on TPU for new checkpoint
   - OR check GCS: gs://layne-tpu-code-sync/checkpoints/vjepa2/

2. CAPTURE W&B metrics over last ~200 steps:
   - Go to W&B dashboard 
   - Export train/lr, train/grad_norm, train/loss
   - Should show: lr=1e-4 (flat), grad_norm>0 (finite), loss trending

3. DOCUMENT throughput baseline:
   - Current: per_core_batch_size=1, grad_accum_steps=4
   - Global batch = 1 * 16_cores * 4_accum = 64
   - Record samples/sec/core from recent logs

4. THEN proceed to Step 2: Verify XLA-friendly metrics

CRITICAL: Do NOT stop training to create snapshot!
The training is stable and we can work around it.
"""
    
    with open(snapshot_dir / "instructions.txt", "w") as f:
        f.write(instructions)
    
    print(f"✅ Baseline snapshot prepared: {snapshot_dir}")
    print("📋 Check instructions.txt for next steps")
    print("🚫 DO NOT stop the running training!")
    
    return snapshot_dir

if __name__ == "__main__":
    snapshot_dir = create_baseline_snapshot()
    print(f"\nSnapshot created: {snapshot_dir}")