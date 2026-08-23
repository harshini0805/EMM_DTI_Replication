"""
Fix all results by populating metrics from CSV files
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def fix_cv_dataset(dataset: str):
    """Fix CV dataset results.json from cv_results.csv"""

    print(f"\n  Processing CV dataset: {dataset}")
    print("  " + "─" * 60)

    results_dir = Path("results") / dataset / "cv_results"
    csv_file = results_dir / "cv_results.csv"
    json_file = Path("results") / dataset / "results.json"

    if not csv_file.exists():
        print(f"  ✗ {csv_file} not found")
        return False

    print(f"  Reading {csv_file.name}...")
    df = pd.read_csv(csv_file)
    print(f"  Found {len(df)} fold results")

    # Map column names
    col_mapping = {
        'roc_auc': 'auc',
        'pr_auc': 'aupr'
    }

    metrics = {}
    for col in df.columns:
        if col in ['fold', 'seed']:
            continue

        # Get the metric name
        metric_name = col_mapping.get(col, col)

        # Calculate mean and std
        values = df[col].dropna()
        if len(values) > 0:
            metrics[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values))
            }
            print(f"    {metric_name:<12}: {metrics[metric_name]['mean']:>8.4f} ± {metrics[metric_name]['std']:.4f}")

    # Update results.json
    results_data = {
        "dataset": dataset,
        "num_seeds": len(df['seed'].unique()),
        "num_folds": 5,
        "total_evaluations": len(df),
        "metrics": metrics
    }

    with open(json_file, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"  ✓ Updated {json_file.name}")
    return True


def fix_independent_dataset(dataset: str):
    """Fix independent dataset summary_results.json from all_runs_results.csv"""

    print(f"\n  Processing independent dataset: {dataset}")
    print("  " + "─" * 60)

    results_dir = Path("results") / f"{dataset}_independent"
    csv_file = results_dir / "all_runs_results.csv"
    json_file = results_dir / "summary_results.json"

    if not csv_file.exists():
        print(f"  ✗ {csv_file} not found")
        return False

    print(f"  Reading {csv_file.name}...")
    df = pd.read_csv(csv_file)
    print(f"  Found {len(df)} run results")

    metrics = {}
    for col in df.columns:
        if col in ['seed', 'run']:
            continue

        # Calculate mean and std
        values = df[col].dropna()
        if len(values) > 0:
            metrics[col] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values))
            }
            print(f"    {col:<12}: {metrics[col]['mean']:>8.4f} ± {metrics[col]['std']:.4f}")

    # Update summary_results.json
    with open(json_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  ✓ Updated {json_file.name}")
    return True


def main():
    print("\n" + "=" * 80)
    print("  FIXING ALL RESULTS.JSON AND SUMMARY_RESULTS.JSON")
    print("=" * 80)

    # CV datasets
    cv_datasets = ["enzyme", "drugbank"]
    for dataset in cv_datasets:
        if (Path("results") / dataset / "cv_results" / "cv_results.csv").exists():
            fix_cv_dataset(dataset)
        else:
            print(f"\n  Skipping {dataset} (no cv_results.csv found)")

    # Independent datasets
    independent_datasets = ["human", "biosnap", "celegans", "bindingdb"]
    for dataset in independent_datasets:
        if (Path("results") / f"{dataset}_independent" / "all_runs_results.csv").exists():
            fix_independent_dataset(dataset)
        else:
            print(f"\n  Skipping {dataset} (no all_runs_results.csv found)")

    print("\n" + "=" * 80)
    print("  ✓ All results fixed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
