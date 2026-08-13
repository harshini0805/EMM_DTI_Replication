"""
Evaluation script for EMM-DTI model.

Usage:
    python -m emm_dti.evaluate --checkpoint results/human/best_model.pt --data_dir data/human
"""

import argparse
import torch
from pathlib import Path
import logging
import numpy as np

from emm_dti.utils.config import Config
from emm_dti.utils.logging_utils import setup_logging
from emm_dti.utils.device import get_device, print_device_info
from emm_dti.data.loaders import DTIDataModule
from emm_dti.models.emm_dti import EMMDTI
from emm_dti.training.trainer import Trainer
from emm_dti.training.metrics import Metrics


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate EMM-DTI model on test set"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/human",
        help="Data directory",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_human.yaml",
        help="Configuration file (for model dimensions)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/evaluation",
        help="Directory to save evaluation results",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default="cuda",
        help="Device to use",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()

    # ===== Setup Logging =====
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(
        log_dir=output_dir,
        level="INFO",
    )

    logger.info("=" * 80)
    logger.info("EMM-DTI Evaluation Script")
    logger.info("=" * 80)

    # ===== Device Setup =====
    device = get_device(args.device)
    print_device_info()

    # ===== Load Configuration =====
    logger.info(f"\nLoading config from: {args.config}")
    config = Config.from_yaml(args.config)

    # ===== Load Data =====
    logger.info(f"Loading data from: {args.data_dir}")
    try:
        data_module = DTIDataModule(
            data_dir=args.data_dir,
            train_split=config.dataset.train_split,
            val_split=config.dataset.val_split,
            test_split=config.dataset.test_split,
        )
    except FileNotFoundError as e:
        logger.error(f"Data loading failed: {e}")
        return 1

    _, _, test_loader = data_module.create_loaders(
        batch_size=args.batch_size,
        num_workers=4,
    )

    logger.info(f"Test set size: {len(test_loader.dataset)}")

    # ===== Initialize Model =====
    logger.info("\nInitializing model...")
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

    # ===== Load Checkpoint =====
    logger.info(f"Loading checkpoint from: {args.checkpoint}")
    try:
        model.load_checkpoint(args.checkpoint)
    except FileNotFoundError as e:
        logger.error(f"Checkpoint not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return 1

    model = model.to(device)

    # ===== Evaluate =====
    logger.info("\nEvaluating on test set...")
    trainer = Trainer(model, device, output_dir=output_dir)

    try:
        predictions, targets = trainer.predict(test_loader)
        predictions = np.array(predictions).squeeze()
        targets = np.array(targets)

        # Compute metrics
        metrics = Metrics.compute_metrics(targets, predictions)

        # Log results
        logger.info("\n" + "=" * 80)
        logger.info("Test Set Results")
        logger.info("=" * 80)
        logger.info(Metrics.format_metrics(metrics, prefix=""))

        # Detailed breakdown
        logger.info("\nDetailed Metrics:")
        logger.info(f"  AUC:         {metrics.get('auc', np.nan):.4f}")
        logger.info(f"  AUPR:        {metrics.get('aupr', np.nan):.4f}")
        logger.info(f"  Accuracy:    {metrics.get('accuracy', np.nan):.4f}")
        logger.info(f"  Precision:   {metrics.get('precision', np.nan):.4f}")
        logger.info(f"  Recall:      {metrics.get('recall', np.nan):.4f}")
        logger.info(f"  Specificity: {metrics.get('specificity', np.nan):.4f}")
        logger.info(f"  F1-Score:    {metrics.get('f1', np.nan):.4f}")

        logger.info("\nConfusion Matrix:")
        logger.info(f"  TP: {metrics.get('tp', 0):5d}  |  FP: {metrics.get('fp', 0):5d}")
        logger.info(f"  FN: {metrics.get('fn', 0):5d}  |  TN: {metrics.get('tn', 0):5d}")

        # Save results
        import json

        results_file = output_dir / "evaluation_results.json"
        results_dict = {k: float(v) if isinstance(v, (np.floating, float)) else v
                       for k, v in metrics.items()}
        with open(results_file, "w") as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"\nResults saved to: {results_file}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
