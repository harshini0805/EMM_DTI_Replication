"""
Training script for EMM-DTI model.

Usage:
    python -m emm_dti.train --config configs/train_human.yaml
    python -m emm_dti.train --config configs/train_human.yaml --epochs 50 --batch_size 64
"""

import argparse
import torch
from pathlib import Path
import logging

from emm_dti.utils.config import Config
from emm_dti.utils.logging_utils import setup_logging
from emm_dti.utils.device import get_device, print_device_info
from emm_dti.data.loaders import DTIDataModule
from emm_dti.models.emm_dti import EMMDTI
from emm_dti.training.trainer import Trainer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train EMM-DTI model for drug-target interaction prediction"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_human.yaml",
        help="Path to configuration YAML file",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Data directory (overrides config)",
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

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def main():
    """Main training function."""
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
    log_dir = Path(config.logging.log_dir)
    logger = setup_logging(
        log_dir=log_dir,
        level=config.logging.log_level,
    )

    logger.info("=" * 80)
    logger.info("EMM-DTI Training Script")
    logger.info("=" * 80)
    logger.info(config)

    # ===== Device Setup =====
    device = get_device(config.training.device)
    print_device_info()

    # ===== Set Random Seeds =====
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    logger.info(f"Random seed: {args.seed}")

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
        logger.error(
            f"Please ensure your data is in {config.dataset.data_dir} with files:\n"
            "  - drugs.csv (columns: drug_id, smiles)\n"
            "  - proteins.csv (columns: protein_id, sequence)\n"
            "  - interactions.csv (columns: drug_id, protein_id, interaction)"
        )
        return 1

    # Create data loaders
    train_loader, val_loader, test_loader = data_module.create_loaders(
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
    )

    logger.info(f"Vocab size: {len(data_module.fcs_vocab)}")

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

    logger.info(model)

    # ===== Initialize Trainer =====
    output_dir = Path(config.logging.log_dir)
    trainer = Trainer(model, device, output_dir=output_dir)

    # ===== Train =====
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

        logger.info("\n" + "=" * 80)
        logger.info("Training completed successfully!")
        logger.info(f"Best model saved to: {output_dir / 'best_model.pt'}")
        logger.info("=" * 80)

        # ===== Test Evaluation =====
        logger.info("\nEvaluating on test set...")
        model.load_checkpoint(output_dir / "best_model.pt")

        from emm_dti.training.metrics import Metrics
        import numpy as np

        predictions, targets = trainer.predict(test_loader)
        predictions = np.array(predictions).squeeze()
        targets = np.array(targets)

        test_metrics = Metrics.compute_metrics(targets, predictions)
        logger.info(f"Test: {Metrics.format_metrics(test_metrics, prefix='test_')}")

        # Save config
        config.save(output_dir / "config.yaml")
        logger.info(f"Config saved to: {output_dir / 'config.yaml'}")

        return 0

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
