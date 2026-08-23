"""
Aggregate existing CV results into results.json format.

For datasets that have already been trained, this script reads the existing
cv_results.csv or cv_summary.csv files and creates/updates results.json
with all metrics including AUC and AUPR.

Usage:
    python aggregate_existing_results.py --dataset enzyme
    python aggregate_existing_results.py --dataset human
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


def aggregate_cv_dataset(dataset: str):
    """Aggregate CV dataset results from csv to json."""
    results_dir = Path("results") / dataset / "cv_results"

    if not results_dir.exists():
        print(f"✗ Results directory not found: {results_dir}")
        return False

    # Try to read cv_results.csv first (has all individual fold results)
    cv_csv = results_dir / "cv_results.csv"
    if not cv_csv.exists():
        print(f"✗ cv_results.csv not found: {cv_csv}")
        return False

    print(f"  Reading {cv_csv}...")
    df = pd.read_csv(cv_csv)

    # Calculate means and stds for each metric
    metrics = {}

    # Map column names - they might be named differently
    metric_names = {
        'auc': ['auc', 'roc_auc', 'val_auc', 'val_roc_auc'],
        'aupr': ['aupr', 'pr_auc', 'val_aupr', 'val_pr_auc'],
        'accuracy': ['accuracy', 'val_accuracy'],
        'precision': ['precision', 'val_precision'],
        'recall': ['recall', 'val_recall'],
        'specificity': ['specificity', 'val_specificity'],
        'mcc': ['mcc', 'val_mcc']
    }

    for metric_key, possible_names in metric_names.items():
        for col_name in possible_names:
            if col_name in df.columns:
                values = df[col_name].dropna()
                if len(values) > 0:
                    metrics[metric_key] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values))
                    }
                    print(f"    {metric_key}: {metrics[metric_key]['mean']:.4f} ± {metrics[metric_key]['std']:.4f}")
                break

    # Create results.json
    num_folds = len(df)
    num_seeds = len(df['seed'].unique()) if 'seed' in df.columns else 1

    results_json_data = {
        "dataset": dataset,
        "num_seeds": num_seeds,
        "num_folds": 5,
        "total_evaluations": num_folds,
        "metrics": metrics
    }

    results_json = results_dir / "results.json"
    with open(results_json, "w") as f:
        json.dump(results_json_data, f, indent=2)

    print(f"  ✓ Saved to {results_json}\n")
    return True


def aggregate_independent_dataset(dataset: str):
    """Aggregate independent dataset results from runs.json to results.json."""
    results_dir = Path("results") / dataset / "cv_results"

    if not results_dir.exists():
        print(f"✗ Results directory not found: {results_dir}")
        return False

    # Try to read runs.json (has all individual run results)
    runs_json = results_dir / "runs.json"
    if not runs_json.exists():
        print(f"✗ runs.json not found: {runs_json}")
        return False

    print(f"  Reading {runs_json}...")
    with open(runs_json, "r") as f:
        all_runs = json.load(f)

    if len(all_runs) == 0:
        print(f"✗ No runs found in {runs_json}")
        return False

    print(f"  Found {len(all_runs)} runs")

    # Aggregate metrics
    all_metrics = {
        "auc": [],
        "aupr": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "specificity": [],
        "mcc": []
    }

    for run in all_runs:
        metrics = run.get("metrics", {})
        for metric_name in all_metrics.keys():
            val = metrics.get(metric_name)
            if val is not None and not np.isnan(float(val)):
                all_metrics[metric_name].append(float(val))

    # Compute summary
    summary_metrics = {}
    for metric_name, values in all_metrics.items():
        if values:
            summary_metrics[metric_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values))
            }
            print(f"    {metric_name}: {summary_metrics[metric_name]['mean']:.4f} ± {summary_metrics[metric_name]['std']:.4f}")

    # Create results.json
    results_json_data = {
        "dataset": dataset,
        "num_seeds": 5,
        "num_folds": 1,
        "total_evaluations": len(all_runs),
        "metrics": summary_metrics
    }

    results_json = results_dir / "results.json"
    with open(results_json, "w") as f:
        json.dump(results_json_data, f, indent=2)

    print(f"  ✓ Saved to {results_json}\n")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate existing results into results.json format"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (enzyme, drugbank, human, biosnap, celegans, bindingdb)",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["cv", "independent", "auto"],
        default="auto",
        help="Dataset type (default: auto-detect)",
    )
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"  Aggregating Results: {args.dataset.upper()}")
    print(f"{'='*80}\n")

    # Auto-detect type if not specified
    dataset_type = args.type
    if dataset_type == "auto":
        cv_datasets = ["enzyme", "drugbank"]
        dataset_type = "cv" if args.dataset in cv_datasets else "independent"
        print(f"  Auto-detected as: {dataset_type}")

    if dataset_type == "cv":
        success = aggregate_cv_dataset(args.dataset)
    else:
        success = aggregate_independent_dataset(args.dataset)

    if success:
        print(f"  ✓ Done! Use: python aggregate_results.py --dataset {args.dataset}")
    else:
        print(f"  ✗ Failed to aggregate results")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
