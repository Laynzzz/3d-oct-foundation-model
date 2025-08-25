#!/usr/bin/env python3
"""
Checkpoint Performance Leaderboard
Analyzes and compares V-JEPA2 checkpoint performance for P1 evaluation.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import wandb


def extract_run_metrics(run_dir: str) -> Dict:
    """Extract final metrics from a training run directory."""
    metrics = {}
    
    # Look for best checkpoint metrics
    best_metrics_file = Path(run_dir) / "best_metrics.json"
    if best_metrics_file.exists():
        with open(best_metrics_file, 'r') as f:
            metrics = json.load(f)
    
    # Extract checkpoint name from path
    checkpoint_name = Path(run_dir).name
    metrics['checkpoint_name'] = checkpoint_name
    
    return metrics


def analyze_wandb_runs(project: str, entity: str, run_names: List[str] = None) -> pd.DataFrame:
    """Extract metrics from W&B runs."""
    api = wandb.Api()
    
    # Get runs from the project
    runs = api.runs(f"{entity}/{project}")
    
    results = []
    for run in runs:
        if run_names and run.name not in run_names:
            continue
            
        # Extract final metrics
        summary = run.summary
        config = run.config
        
        result = {
            'run_name': run.name,
            'checkpoint_path': config.get('paths.checkpoint_path', 'unknown'),
            'checkpoint_name': extract_checkpoint_name(config.get('paths.checkpoint_path', '')),
            'final_train_loss': summary.get('train/loss', None),
            'final_val_loss': summary.get('val/loss', None),
            'best_val_accuracy': summary.get('val/accuracy', None),
            'best_balanced_accuracy': summary.get('val/balanced_accuracy', None),
            'best_macro_f1': summary.get('val/macro_f1', None),
            'best_auroc': summary.get('val/auroc_macro', None),
            'total_epochs': summary.get('epoch', None),
            'training_time_hours': summary.get('_runtime', 0) / 3600,
            'state': run.state
        }
        results.append(result)
    
    return pd.DataFrame(results)


def extract_checkpoint_name(checkpoint_path: str) -> str:
    """Extract checkpoint name from path."""
    if 'multi_domain' in checkpoint_path:
        return 'multi_domain'
    elif 'single_domain_01' in checkpoint_path:
        return 'single_domain_01'
    elif 'single_domain_02' in checkpoint_path:
        return 'single_domain_02'
    else:
        return 'unknown'


def create_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    """Create performance leaderboard ranked by balanced accuracy."""
    # Filter completed runs
    completed_df = df[df['state'] == 'finished'].copy()
    
    if completed_df.empty:
        print("No completed runs found!")
        return df
    
    # Rank by balanced accuracy (primary metric)
    completed_df = completed_df.sort_values('best_balanced_accuracy', ascending=False)
    completed_df['rank'] = range(1, len(completed_df) + 1)
    
    # Select key columns for leaderboard
    leaderboard_cols = [
        'rank', 'checkpoint_name', 'best_balanced_accuracy', 
        'best_val_accuracy', 'best_macro_f1', 'best_auroc',
        'final_val_loss', 'total_epochs', 'training_time_hours'
    ]
    
    return completed_df[leaderboard_cols]


def print_leaderboard(df: pd.DataFrame):
    """Print formatted leaderboard."""
    print("\n🏆 V-JEPA2 Checkpoint Performance Leaderboard")
    print("=" * 80)
    print(f"{'Rank':<4} {'Checkpoint':<18} {'Bal_Acc':<8} {'Accuracy':<8} {'F1':<6} {'AUROC':<6} {'Val_Loss':<8} {'Epochs':<7} {'Time_h':<6}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        print(f"{int(row['rank']):<4} "
              f"{row['checkpoint_name']:<18} "
              f"{row['best_balanced_accuracy']:.4f}   "
              f"{row['best_val_accuracy']:.4f}   "
              f"{row['best_macro_f1']:.4f} "
              f"{row['best_auroc']:.4f} "
              f"{row['final_val_loss']:.4f}   "
              f"{int(row['total_epochs']):<7} "
              f"{row['training_time_hours']:.1f}")


def generate_summary_report(df: pd.DataFrame) -> str:
    """Generate summary analysis report."""
    if df.empty:
        return "No completed runs to analyze."
    
    best_row = df.iloc[0]
    
    report = f"""
## V-JEPA2 Checkpoint Evaluation Summary

### 🥇 Best Performing Checkpoint: {best_row['checkpoint_name']}
- **Balanced Accuracy**: {best_row['best_balanced_accuracy']:.4f}
- **Standard Accuracy**: {best_row['best_val_accuracy']:.4f}
- **Macro F1 Score**: {best_row['best_macro_f1']:.4f}
- **AUROC**: {best_row['best_auroc']:.4f}
- **Training Time**: {best_row['training_time_hours']:.1f} hours

### 📊 Performance Analysis
- **Total Checkpoints Evaluated**: {len(df)}
- **Best Balanced Accuracy**: {df['best_balanced_accuracy'].max():.4f}
- **Average Balanced Accuracy**: {df['best_balanced_accuracy'].mean():.4f}
- **Performance Spread**: {df['best_balanced_accuracy'].max() - df['best_balanced_accuracy'].min():.4f}

### 🎯 Recommendations
1. **Best for Production**: {best_row['checkpoint_name']} checkpoint
2. **Next Steps**: Fine-tune the winning checkpoint with unfrozen encoder
3. **Architecture Insights**: Multi-domain vs single-domain performance comparison
"""
    
    return report


def main():
    parser = argparse.ArgumentParser(description='Generate V-JEPA2 checkpoint leaderboard')
    parser.add_argument('--wandb-project', default='3d-oct-foundation-model')
    parser.add_argument('--wandb-entity', default='laynzzz-university-at-buffalo')
    parser.add_argument('--output-file', default='checkpoint_leaderboard.csv')
    parser.add_argument('--runs-dir', help='Local runs directory (alternative to W&B)')
    
    args = parser.parse_args()
    
    print("🔍 Extracting checkpoint performance metrics...")
    
    # Extract metrics from W&B
    try:
        df = analyze_wandb_runs(args.wandb_project, args.wandb_entity)
    except Exception as e:
        print(f"Failed to fetch W&B data: {e}")
        df = pd.DataFrame()
    
    if df.empty:
        print("No runs found. Make sure training runs have completed.")
        return
    
    # Create leaderboard
    leaderboard = create_leaderboard(df)
    
    if leaderboard.empty:
        print("No completed runs found for leaderboard.")
        return
    
    # Display results
    print_leaderboard(leaderboard)
    
    # Generate summary report
    report = generate_summary_report(leaderboard)
    print(report)
    
    # Save results
    leaderboard.to_csv(args.output_file, index=False)
    print(f"\n💾 Leaderboard saved to: {args.output_file}")
    
    # Save report
    report_file = args.output_file.replace('.csv', '_report.md')
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"📝 Report saved to: {report_file}")


if __name__ == "__main__":
    main()