"""
Cross-Validation Training Script for EMM-DTI

For datasets that support stratified CV (enzyme, drugbank):
- 5-fold stratified cross-validation
- 3 random seeds (42, 123, 2024)
- Total: 15 results per dataset

Usage:
    python -m emm_dti.train_cv --data_dir data/enzyme --epochs 200
    python -m emm_dti.train_cv --data_dir data/drugbank --cv_folds 5 --seeds 42 123 2024
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Dict, List

from emm_dti.utils.config import Config
from emm_dti.utils.logging_utils import setup_logging
from emm_dti.utils.device import get_device, print_device_info
from emm_dti.data.loaders import DTIDataModule
from emm_dti.models.emm_dti import EMMDTI
from emm_dti.training.trainer import Trainer
from emm_dti.training.metrics import Metrics

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train EMM-DTI with K-fold stratified cross-validation"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_enzyme.yaml",
        help="Path to configuration YAML file",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Data directory (overrides config)",
    )

    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5)",
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 2024],
        help="Random seeds for CV folds (default: 42 123 2024)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device to use (overrides config)",
    )

    return parser.parse_args()


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def perform_cv(
    data_module: DTIDataModule,
    num_folds: int,
    seeds: List[int],
    config: Config,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, List[float]]:
    """
    Perform k-fold stratified cross-validation.

    Returns:
        Dictionary with metrics for each fold and seed
    """
    all_results = {
        "fold": [],
        "seed": [],
        "accuracy": [],
        "precision": [],
        "recall": [],
        "specificity": [],
        "mcc": [],
        "roc_auc": [],
        "pr_auc": [],
    }

    fold_idx = 0

    for seed in seeds:
        set_seed(seed)

        logger.info("\n" + "=" * 80)
        logger.info(f"SEED {seeds.index(seed) + 1}/{len(seeds)}: seed={seed}")
        logger.info("=" * 80)

        # Get fold indices (stratified)
        fold_results = data_module.get_stratified_folds(
            num_folds=num_folds,
            random_state=seed,
        )

        for fold_num, (train_idx, val_idx, test_idx) in enumerate(fold_results):
            fold_idx += 1
            logger.info(f"\n--- Fold {fold_num + 1}/{num_folds} ---")

            # Create fold-specific data loaders
            train_loader = data_module.create_fold_loader(
                train_idx,
                batch_size=config.training.batch_size,
                num_workers=config.training.num_workers,
            )
            val_loader = data_module.create_fold_loader(
                val_idx,
                batch_size=config.training.batch_size,
                num_workers=config.training.num_workers,
            )
            test_loader = data_module.create_fold_loader(
                test_idx,
                batch_size=config.training.batch_size,
                num_workers=config.training.num_workers,
            )

            logger.info(
                f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}"
            )

            # Initialize model
            model = EMMDTI(
                vocab_size=len(data_module.fcs_vocab),
                fcs_embedding_dim=config.model.fcs_embedding_dim,
                mamba_hidden_dim=config.model.mamba_hidden_dim,
                mamba_n_layers=config.model.mamba_n_layers,
                mamba_state_size=config.model.mamba_state_size,
                mamba_expand_factor=config.model.mamba_expand_factor,
                cnn_out_channels=config.model.cnn_out_channels,
                cnn_kernel_size=config.model.cnn_kernel_size,
                dropout=config.model.dropout,
            )

            # Setup trainer
            fold_output_dir = output_dir / f"fold_{fold_num}_seed_{seed}"
            fold_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Checkpoint directory: {fold_output_dir}")

            trainer = Trainer(model, device, output_dir=fold_output_dir)

            # Train
            try:
                history = trainer.fit(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=config.training.epochs,
                    learning_rate=config.training.learning_rate,
                    optimizer_name=config.optimization.optimizer,
                    weight_decay=config.optimization.weight_decay,
                    gradient_clip=config.optimization.gradient_clip,
                    early_stopping_patience=config.training.early_stopping_patience,
                    scheduler_name=config.optimization.lr_scheduler,
                )

                logger.info(
                    f"✓ Training completed (best epoch: {trainer.best_epoch})"
                )

            except Exception as e:
                logger.error(f"✗ Training failed: {e}", exc_info=True)
                continue

            # Evaluate on test set
            checkpoint_path = fold_output_dir / "best_model.pt"

            # Check if checkpoint exists
            if not checkpoint_path.exists():
                logger.error(f"✗ Checkpoint not found: {checkpoint_path}")
                logger.error(f"   This usually means training failed or no improvement occurred.")
                logger.error(f"   Check training logs above for details.")
                continue

            try:
                model.load_checkpoint(checkpoint_path)
                logger.info(f"✓ Loaded checkpoint from {checkpoint_path}")

                predictions, targets = trainer.predict(test_loader)
                predictions = np.array(predictions).squeeze()
                targets = np.array(targets)

                # Apply sigmoid to convert logits to probabilities
                predictions = 1.0 / (1.0 + np.exp(-predictions))

                # Compute metrics
                metrics = Metrics.compute_metrics(targets, predictions)

                logger.info(
                    f"✓ Test Metrics: "
                    f"AUC={metrics['roc_auc']:.4f} | "
                    f"AUPR={metrics['aupr']:.4f} | "
                    f"Acc={metrics['accuracy']:.4f}"
                )

                # Store results
                all_results["fold"].append(fold_num + 1)
                all_results["seed"].append(seed)
                all_results["accuracy"].append(metrics["accuracy"])
                all_results["precision"].append(metrics["precision"])
                all_results["recall"].append(metrics["recall"])
                all_results["specificity"].append(metrics["specificity"])
                all_results["mcc"].append(metrics["mcc"])
                all_results["roc_auc"].append(metrics["roc_auc"])
                all_results["pr_auc"].append(metrics["aupr"])

            except Exception as e:
                logger.error(f"✗ Evaluation failed: {e}", exc_info=True)
                continue

    return all_results


def main():
    """Main CV training function."""
    args = parse_args()

    # ===== Load Configuration =====
    config = Config.from_yaml(args.config)

    # Override config with command-line args
    if args.data_dir:
        config.dataset.data_dir = args.data_dir
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.epochs:
        config.training.epochs = args.epochs
    if args.device:
        config.training.device = args.device

    # ===== Setup Logging =====
    log_dir = Path(config.logging.log_dir) / "cv_results"
    logger_inst = setup_logging(
        log_dir=log_dir,
        level=config.logging.log_level,
    )

    logger.info("=" * 80)
    logger.info("EMM-DTI Cross-Validation Training")
    logger.info("=" * 80)
    logger.info(config)

    # ===== Device Setup =====
    device = get_device(config.training.device)
    print_device_info()

    # ===== Load Data =====
    logger.info("\nLoading data...")
    try:
        data_module = DTIDataModule(
            data_dir=config.dataset.data_dir,
            train_split=config.dataset.train_split,
            val_split=config.dataset.val_split,
            test_split=config.dataset.test_split,
            random_seed=config.dataset.random_seed,
        )
    except FileNotFoundError as e:
        logger.error(f"Data loading failed: {e}")
        return 1

    logger.info(f"✓ Data loaded. Vocab size: {len(data_module.fcs_vocab)}")

    # ===== Cross-Validation =====
    output_dir = Path(config.logging.log_dir) / "cv_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    cv_results = perform_cv(
        data_module=data_module,
        num_folds=args.cv_folds,
        seeds=args.seeds,
        config=config,
        device=device,
        output_dir=output_dir,
    )

    # ===== Summary Statistics =====
    logger.info("\n" + "=" * 80)
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info("=" * 80)

    if cv_results["roc_auc"]:
        roc_auc_mean = np.mean(cv_results["roc_auc"])
        roc_auc_std = np.std(cv_results["roc_auc"])
        aupr_mean = np.mean(cv_results["pr_auc"])
        aupr_std = np.std(cv_results["pr_auc"])
        acc_mean = np.mean(cv_results["accuracy"])
        acc_std = np.std(cv_results["accuracy"])

        logger.info(f"Total folds completed: {len(cv_results['roc_auc'])}")
        logger.info(f"\nROC-AUC:  {roc_auc_mean:.4f} ± {roc_auc_std:.4f}")
        logger.info(f"AUPR:     {aupr_mean:.4f} ± {aupr_std:.4f}")
        logger.info(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")

        logger.info(f"\nDetailed results saved to: {output_dir}")

    logger.info("=" * 80)
    return 0


if __name__ == "__main__":
    exit(main())
