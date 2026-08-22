"""
Multi-seed ensemble training for EMM-DTI.

Trains independent runs with different random seeds, averages predictions,
and reports metrics with error bars (mean ± std).

Usage:
    python ensemble_train.py

Seeds: [42, 123, 2024] (same as reference implementations)
Per-run: 200 epochs, 30 patience, then evaluate on test set
"""

import os
import sys
import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from emm_dti.models.emm_dti import EMMDTI
from emm_dti.data.loaders import DTIDataModule
from emm_dti.training.trainer import Trainer
from emm_dti.training.metrics import Metrics
from emm_dti.utils.device import get_device
from emm_dti.utils.logging_utils import setup_logging

# ============================================================
# CONFIGURATION
# ============================================================

SEEDS = [42, 123, 2024, 456, 789]  # 5 different seeds, 5 runs total
DATA_DIR = "data/human"
CONFIG_PATH = "configs/train_human.yaml"
ENSEMBLE_DIR = Path("results/ensemble")
EPOCHS = 200
PATIENCE = 30
BATCH_SIZE = 32

# ============================================================
# REPRODUCIBILITY
# ============================================================


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# DEVICE SETUP
# ============================================================

device = get_device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# LOAD DATA (ONCE, shared across all seeds)
# ============================================================

print("\n" + "=" * 80)
print("Loading data...")
print("=" * 80)

try:
    data_module = DTIDataModule(
        data_dir=DATA_DIR,
        train_split=0.7,
        val_split=0.2,
        test_split=0.1,
        random_seed=42,  # Fixed seed for data loading (not model training)
    )

    train_loader, val_loader, test_loader = data_module.create_loaders(
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    print(f"Vocab size: {len(data_module.fcs_vocab)}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

except Exception as e:
    print(f"ERROR loading data: {e}")
    sys.exit(1)

# ============================================================
# TRAIN ONE SEED
# ============================================================


def train_seed(seed, seed_index):
    """
    Train one complete run with the given seed.

    Returns:
        - model: trained model
        - test_predictions: numpy array of test predictions
        - test_targets: numpy array of test targets
    """

    print(f"\n{'#' * 80}")
    print(f"# SEED {seed_index + 1}/{len(SEEDS)}: seed={seed}")
    print(f"{'#' * 80}")

    set_seed(seed)

    # ===== Initialize Model =====
    model = EMMDTI(
        vocab_size=len(data_module.fcs_vocab),
        fcs_embedding_dim=128,
        mamba_hidden_dim=64,  # Standardized to match other 9 architectures
        mamba_n_layers=2,
        mamba_state_size=16,
        mamba_expand_factor=2,
        cnn_out_channels=3,
        cnn_kernel_size=3,
        dropout=0.1,
    )

    # ===== Setup Trainer =====
    seed_output_dir = ENSEMBLE_DIR / f"seed_{seed}"
    seed_output_dir.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(model, device, output_dir=seed_output_dir)

    # ===== Train =====
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = trainer.setup_optimizer(
        learning_rate=0.001,
        optimizer_name="adam",
        weight_decay=1e-5,
    )

    print(f"\nTraining for up to {EPOCHS} epochs (patience={PATIENCE})...")

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        learning_rate=0.001,
        optimizer_name="adam",
        weight_decay=1e-5,
        gradient_clip=1.0,
        early_stopping_patience=PATIENCE,
        scheduler_name="none",
    )

    # ===== Load Best Model & Evaluate =====
    best_model_path = seed_output_dir / "best_model.pt"
    model.load_checkpoint(best_model_path)

    print(f"\nEvaluating on test set...")
    test_predictions, test_targets = trainer.predict(test_loader)

    test_predictions = np.array(test_predictions).squeeze()
    test_targets = np.array(test_targets)

    test_metrics = Metrics.compute_metrics(test_targets, test_predictions)

    print(f"\n[Seed {seed}] Test Metrics:")
    print(Metrics.format_metrics(test_metrics, prefix=""))

    # Save per-seed results
    seed_results = {
        "seed": seed,
        "metrics": {k: float(v) if not np.isnan(v) else None for k, v in test_metrics.items()},
        "predictions_shape": test_predictions.shape,
        "targets_shape": test_targets.shape,
    }

    with open(seed_output_dir / "results.json", "w") as f:
        json.dump(seed_results, f, indent=2)

    return model, test_predictions, test_targets


# ============================================================
# TRAIN ALL SEEDS
# ============================================================

ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)

per_seed_predictions = []
reference_targets = None
per_seed_test_metrics = []

for seed_idx, seed in enumerate(SEEDS):
    try:
        model, test_preds, test_targets = train_seed(seed, seed_idx)

        # Verify targets are identical across seeds
        if reference_targets is None:
            reference_targets = test_targets
        else:
            assert np.allclose(
                test_targets, reference_targets
            ), "Test target order differs between seeds!"

        per_seed_predictions.append(test_preds)

        # Compute metrics for this seed
        seed_metrics = Metrics.compute_metrics(test_targets, test_preds)
        per_seed_test_metrics.append(seed_metrics)

    except Exception as e:
        print(f"\nERROR training seed {seed}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

# ============================================================
# ENSEMBLE: Average Predictions Across Seeds
# ============================================================

print("\n" + "=" * 80)
print("ENSEMBLING PREDICTIONS")
print("=" * 80)

ensemble_predictions = np.mean(np.stack(per_seed_predictions, axis=0), axis=0)
ensemble_metrics = Metrics.compute_metrics(reference_targets, ensemble_predictions)

# ============================================================
# RESULTS SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("PER-SEED TEST METRICS")
print("=" * 80)

metric_names = list(per_seed_test_metrics[0].keys())

for metric_name in metric_names:
    if np.isnan(per_seed_test_metrics[0].get(metric_name, np.nan)):
        continue

    values = [m.get(metric_name, np.nan) for m in per_seed_test_metrics]
    values = [v for v in values if not np.isnan(v)]

    if len(values) > 0:
        mean_val = np.mean(values)
        std_val = np.std(values)

        print(f"\n{metric_name.upper()}")
        for seed, val in zip(SEEDS, values):
            print(f"  seed {seed:<6}: {val:.4f}")
        print(f"  mean ± std: {mean_val:.4f} ± {std_val:.4f}")

print("\n" + "=" * 80)
print("ENSEMBLE TEST METRICS (averaged predictions)")
print("=" * 80)
print(Metrics.format_metrics(ensemble_metrics, prefix=""))

print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'Seed':<8} {'AUC':<10} {'AUPR':<10} {'Accuracy':<10} {'MCC':<10}")
print("-" * 48)
for seed, metrics in zip(SEEDS, per_seed_test_metrics):
    print(
        f"{seed:<8} "
        f"{metrics.get('auc', np.nan):<10.4f} "
        f"{metrics.get('aupr', np.nan):<10.4f} "
        f"{metrics.get('accuracy', np.nan):<10.4f} "
        f"{metrics.get('mcc', np.nan):<10.4f}"
    )

print("-" * 48)
print(
    f"{'ENSEMBLE':<8} "
    f"{ensemble_metrics.get('auc', np.nan):<10.4f} "
    f"{ensemble_metrics.get('aupr', np.nan):<10.4f} "
    f"{ensemble_metrics.get('accuracy', np.nan):<10.4f} "
    f"{ensemble_metrics.get('mcc', np.nan):<10.4f}"
)

# Save ensemble results
ensemble_results = {
    "seeds": SEEDS,
    "ensemble_metrics": {
        k: float(v) if not np.isnan(v) else None for k, v in ensemble_metrics.items()
    },
    "per_seed_metrics": [
        {k: float(v) if not np.isnan(v) else None for k, v in m.items()}
        for m in per_seed_test_metrics
    ],
}

with open(ENSEMBLE_DIR / "ensemble_results.json", "w") as f:
    json.dump(ensemble_results, f, indent=2)

print(f"\nResults saved to: {ENSEMBLE_DIR}")
print("=" * 80)
