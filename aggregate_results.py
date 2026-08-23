"""
Aggregate results for a single dataset (like mamba-dti's compare_architectures.py).
Loads results from CV or independent runs.

Usage:
    python aggregate_results.py --dataset enzyme
    python aggregate_results.py --dataset human
"""
import argparse
import json
from pathlib import Path
import numpy as np

def load_dataset_results(dataset: str) -> dict:
    """Load results.json for a given dataset (CV or independent)."""

    # Try CV dataset format: results/{dataset}/results.json
    cv_path = Path("results") / dataset / "results.json"
    if cv_path.exists():
        try:
            with open(cv_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"  Error loading {dataset}: {e}")
            return None

    # Try independent dataset format: results/{dataset}_independent/summary_results.json
    independent_path = Path("results") / f"{dataset}_independent" / "summary_results.json"
    if independent_path.exists():
        try:
            with open(independent_path, "r") as f:
                metrics = json.load(f)
            # Wrap in standard format
            return {
                "dataset": dataset,
                "num_seeds": 5,
                "num_folds": 1,
                "total_evaluations": 5,
                "metrics": metrics
            }
        except Exception as e:
            print(f"  Error loading {dataset}: {e}")
            return None

    return None

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate results for a single dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (enzyme, drugbank, human, biosnap, celegans, bindingdb)",
    )
    args = parser.parse_args()

    dataset = args.dataset

    print(f"\n{'='*80}")
    print(f"  EMM-DTI Results: {dataset.upper()}")
    print(f"{'='*80}\n")

    print(f"  Loading {dataset}...", end=" ")
    results = load_dataset_results(dataset)

    if not results:
        print("✗ (not found)")
        print(f"\n  Run training first:\n")
        if dataset in ["enzyme", "drugbank"]:
            print(f"    python -m emm_dti.train_cv --data_dir data/{dataset} --epochs 200")
        else:
            print(f"    for seed in 42 123 2024 456 789; do")
            print(f"      python -m emm_dti.train --data_dir data/{dataset} --epochs 200 --seed $seed")
            print(f"    done")
        return

    print("✓\n")

    # Extract metrics
    metrics = results.get("metrics", {})
    total_evals = results.get("total_evaluations", 0)

    print(f"  Total Evaluations: {total_evals}\n")
    print(f"  {'Metric':<15}  {'Mean':>10}  {'Std Dev':>10}")
    print(f"  {'─'*15}  {'─'*10}  {'─'*10}")

    for metric_name in ["auc", "aupr", "accuracy"]:
        if metric_name in metrics:
            metric_data = metrics[metric_name]
            mean_val = metric_data.get("mean", np.nan)
            std_val = metric_data.get("std", np.nan)

            if not np.isnan(mean_val):
                print(f"  {metric_name.upper():<15}  {mean_val:>10.4f}  {std_val:>10.4f}")

    print(f"  {'─'*15}  {'─'*10}  {'─'*10}\n")

if __name__ == "__main__":
    main()
