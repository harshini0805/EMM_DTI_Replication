"""
Fix summary_results.json by adding AUC and AUPR from all_runs_results.csv
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


def fix_summary(dataset: str):
    """Read all_runs_results.csv and create proper summary with AUC and AUPR."""

    logs_dir = Path("logs") / dataset

    # Check for independent dataset structure
    independent_dir = logs_dir / f"{dataset}_independent"

    if not independent_dir.exists():
        print(f"✗ Directory not found: {independent_dir}")
        return False

    all_runs_csv = independent_dir / "all_runs_results.csv"

    if not all_runs_csv.exists():
        print(f"✗ File not found: {all_runs_csv}")
        return False

    print(f"  Reading {all_runs_csv}...")
    df = pd.read_csv(all_runs_csv)

    print(f"  Found {len(df)} results")
    print(f"  Columns: {list(df.columns)}\n")

    # Map metric names - they might have different prefixes
    metrics = {}

    metric_mappings = {
        'auc': ['auc', 'roc_auc', 'test_auc', 'test_roc_auc'],
        'aupr': ['aupr', 'pr_auc', 'test_aupr', 'test_pr_auc'],
        'accuracy': ['accuracy', 'test_accuracy'],
        'precision': ['precision', 'test_precision'],
        'recall': ['recall', 'test_recall'],
        'specificity': ['specificity', 'test_specificity'],
        'mcc': ['mcc', 'test_mcc']
    }

    for metric_key, possible_names in metric_mappings.items():
        for col_name in possible_names:
            if col_name in df.columns:
                values = df[col_name].dropna()
                if len(values) > 0:
                    mean_val = float(np.mean(values))
                    std_val = float(np.std(values))
                    metrics[metric_key] = {
                        "mean": mean_val,
                        "std": std_val
                    }
                    print(f"  {metric_key.upper():<12}: {mean_val:.4f} ± {std_val:.4f}")
                break

    # Save updated summary
    summary_json_path = logs_dir / "summary_results.json"

    with open(summary_json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  ✓ Updated: {summary_json_path}\n")

    return True


def main():
    parser = argparse.ArgumentParser(description="Fix summary_results.json with AUC/AUPR")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (human, biosnap, etc)")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"  Fixing Summary Results: {args.dataset.upper()}")
    print(f"{'='*80}\n")

    if fix_summary(args.dataset):
        print(f"  ✓ Done!")
    else:
        print(f"  ✗ Failed")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
