"""
Training loop for EMM-DTI model.

Handles training, validation, and model checkpointing.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging
from tqdm import tqdm

from emm_dti.models.emm_dti import EMMDTI
from emm_dti.training.metrics import Metrics, MetricsTracker
from emm_dti.utils.device import empty_cuda_cache

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training manager for EMM-DTI model.

    Handles forward/backward passes, validation, metrics computation,
    and checkpoint saving.
    """

    def __init__(
        self,
        model: EMMDTI,
        device: torch.device,
        output_dir: str | Path = "results",
    ):
        """
        Initialize trainer.

        Args:
            model: EMM-DTI model instance
            device: Device to train on (cuda or cpu)
            output_dir: Directory for saving checkpoints and logs
        """
        self.model = model.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.best_auc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()

        logger.info(f"Trainer initialized. Output dir: {self.output_dir}")

    def setup_optimizer(
        self,
        learning_rate: float = 0.001,
        optimizer_name: str = "adam",
        weight_decay: float = 1e-5,
    ) -> torch.optim.Optimizer:
        """
        Setup optimizer.

        Args:
            learning_rate: Learning rate
            optimizer_name: Optimizer type (adam, sgd, adamw)
            weight_decay: Weight decay coefficient

        Returns:
            Optimizer instance
        """
        if optimizer_name.lower() == "adam":
            optimizer = Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_name.lower() == "adamw":
            optimizer = AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_name.lower() == "sgd":
            optimizer = SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        logger.info(f"Optimizer: {optimizer_name}, LR: {learning_rate}")
        return optimizer

    def setup_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler_name: str = "none",
        epochs: int = 100,
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Setup learning rate scheduler.

        Args:
            optimizer: Optimizer instance
            scheduler_name: Scheduler type (none, cosine, linear)
            epochs: Total number of epochs

        Returns:
            Scheduler instance or None
        """
        if scheduler_name.lower() == "none":
            return None
        elif scheduler_name.lower() == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_name.lower() == "linear":
            scheduler = LinearLR(
                optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=epochs,
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        logger.info(f"Learning rate scheduler: {scheduler_name}")
        return scheduler

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        gradient_clip: float = 1.0,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer instance
            loss_fn: Loss function
            gradient_clip: Gradient clipping value

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        all_targets = []
        all_predictions = []

        pbar = tqdm(train_loader, desc="Training")

        for batch_idx, (drug_indices, protein_indices, labels) in enumerate(pbar):
            # Move to device
            drug_indices = drug_indices.to(self.device)
            protein_indices = protein_indices.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            predictions = self.model(drug_indices, protein_indices)
            # Squeeze only the last dimension (output dim), keep batch dim
            loss = loss_fn(predictions.squeeze(-1), labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)

            optimizer.step()

            # Accumulate metrics
            total_loss += loss.item()
            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.detach().cpu().numpy())

            pbar.set_postfix({"loss": loss.item()})

        # Compute metrics (convert logits to probabilities)
        import numpy as np

        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions).squeeze()
        # Apply sigmoid to convert logits to probabilities
        all_predictions = 1.0 / (1.0 + np.exp(-all_predictions))

        metrics = Metrics.compute_metrics(all_targets, all_predictions)
        metrics["loss"] = total_loss / len(train_loader)

        return metrics

    def validate(
        self,
        val_loader: DataLoader,
        loss_fn: nn.Module,
    ) -> Dict[str, float]:
        """
        Validate on validation set.

        Args:
            val_loader: Validation data loader
            loss_fn: Loss function

        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating")

            for drug_indices, protein_indices, labels in pbar:
                # Move to device
                drug_indices = drug_indices.to(self.device)
                protein_indices = protein_indices.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                predictions = self.model(drug_indices, protein_indices)
                # Squeeze only the last dimension (output dim), keep batch dim
                loss = loss_fn(predictions.squeeze(-1), labels)

                total_loss += loss.item()
                all_targets.extend(labels.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

        # Compute metrics (convert logits to probabilities)
        import numpy as np

        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions).squeeze()
        # Apply sigmoid to convert logits to probabilities
        all_predictions = 1.0 / (1.0 + np.exp(-all_predictions))

        metrics = Metrics.compute_metrics(all_targets, all_predictions)
        metrics["loss"] = total_loss / len(val_loader)

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 0.001,
        optimizer_name: str = "adam",
        weight_decay: float = 1e-5,
        gradient_clip: float = 1.0,
        early_stopping_patience: int = 10,
        scheduler_name: str = "none",
    ) -> Dict[str, list]:
        """
        Train model for multiple epochs.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            learning_rate: Learning rate
            optimizer_name: Optimizer type
            weight_decay: Weight decay
            gradient_clip: Gradient clipping value
            early_stopping_patience: Patience for early stopping
            scheduler_name: Learning rate scheduler

        Returns:
            Dictionary with training history
        """
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info("=" * 60)

        # Setup optimizer and scheduler
        optimizer = self.setup_optimizer(learning_rate, optimizer_name, weight_decay)
        scheduler = self.setup_scheduler(optimizer, scheduler_name, epochs)

        # Loss function (numerically stable)
        loss_fn = nn.BCEWithLogitsLoss()

        # Training loop
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")

            # Train
            train_metrics = self.train_epoch(
                train_loader, optimizer, loss_fn, gradient_clip
            )
            self.train_metrics.update(train_metrics)

            # Validate
            val_metrics = self.validate(val_loader, loss_fn)
            self.val_metrics.update(val_metrics)

            # Log metrics
            logger.info(f"Train: {Metrics.format_metrics(train_metrics, prefix='train_')}")
            logger.info(f"Val:   {Metrics.format_metrics(val_metrics, prefix='val_')}")

            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step()

            # Checkpointing
            current_auc = val_metrics.get("auc", 0.0)
            if current_auc > self.best_auc:
                self.best_auc = current_auc
                self.best_epoch = epoch
                self.patience_counter = 0

                # Save best model
                checkpoint_path = self.output_dir / "best_model.pt"
                self.model.save_checkpoint(checkpoint_path, optimizer.state_dict())
                logger.info(f"Saved best model (AUC: {current_auc:.4f})")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                logger.info(
                    f"Early stopping at epoch {epoch + 1} "
                    f"(best epoch: {self.best_epoch + 1})"
                )
                break

            # Memory cleanup
            empty_cuda_cache()

        logger.info("=" * 60)
        logger.info(f"Training complete. Best AUC: {self.best_auc:.4f}")
        logger.info("=" * 60)

        return {
            "train": self.train_metrics.history,
            "val": self.val_metrics.history,
        }

    def predict(
        self,
        test_loader: DataLoader,
    ) -> Tuple[list, list]:
        """
        Make predictions on test set.

        Args:
            test_loader: Test data loader

        Returns:
            Tuple of (predictions as probabilities, targets)
        """
        import numpy as np

        self.model.eval()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for drug_indices, protein_indices, labels in test_loader:
                drug_indices = drug_indices.to(self.device)
                protein_indices = protein_indices.to(self.device)

                logits = self.model(drug_indices, protein_indices)
                # Convert logits to probabilities
                predictions = torch.sigmoid(logits)
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.numpy())

        return all_predictions, all_targets
