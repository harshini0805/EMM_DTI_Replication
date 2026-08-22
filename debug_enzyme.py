#!/usr/bin/env python3
"""
Debug script for EMM-DTI enzyme training.
Runs training with full logging and saves output to file.
"""

import sys
import torch
from pathlib import Path
from datetime import datetime

# Setup paths
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from emm_dti.utils.logging_utils import setup_logging
from emm_dti.utils.device import get_device, print_device_info
from emm_dti.data.loaders import DTIDataModule
from emm_dti.models.emm_dti import EMMDTI
from emm_dti.training.trainer import Trainer


def debug_training():
    """Run debug training on enzyme dataset."""

    # Setup logging
    log_dir = Path(PROJECT_ROOT) / "debug_logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_dir=log_dir, level="DEBUG")

    logger.info("="*80)
    logger.info("EMM-DTI DEBUG TRAINING - ENZYME DATASET")
    logger.info("="*80)

    # Device setup
    device = get_device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    print_device_info()

    # Set seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Load data
    logger.info("\n" + "="*80)
    logger.info("STEP 1: LOADING DATA")
    logger.info("="*80)

    data_dir = Path(PROJECT_ROOT) / "data" / "enzyme"
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Data directory exists: {data_dir.exists()}")

    if data_dir.exists():
        logger.info(f"Files in directory:")
        for f in data_dir.glob("*.csv"):
            logger.info(f"  - {f.name}: {sum(1 for _ in open(f)) - 1} rows")

    try:
        data_module = DTIDataModule(
            data_dir=str(data_dir),
            train_split=0.7,
            val_split=0.2,
            test_split=0.1,
            random_seed=42,
        )
        logger.info(f"✓ Data loaded successfully")
        logger.info(f"  Vocab size: {len(data_module.fcs_vocab)}")
        fcs_patterns = data_module.fcs.get_patterns()
        logger.info(f"  FCS patterns: {sum(len(p) for p in fcs_patterns.values())}")
        logger.info(f"  FCS patterns by length: {[(k, len(v)) for k, v in sorted(fcs_patterns.items())]}")

    except Exception as e:
        logger.error(f"✗ Data loading failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return

    # Create loaders
    logger.info("\n" + "="*80)
    logger.info("STEP 2: CREATING DATA LOADERS")
    logger.info("="*80)

    train_loader, val_loader, test_loader = data_module.create_loaders(
        batch_size=16,
        num_workers=0,
        pin_memory=False,
    )

    logger.info(f"Train loader: {len(train_loader)} batches")
    logger.info(f"Val loader: {len(val_loader)} batches")
    logger.info(f"Test loader: {len(test_loader)} batches")

    # Sample batch
    logger.info("\nSample batch from training set:")
    sample_batch = next(iter(train_loader))
    logger.info(f"  Drug indices shape: {sample_batch[0].shape}")
    logger.info(f"  Protein indices shape: {sample_batch[1].shape}")
    logger.info(f"  Labels shape: {sample_batch[2].shape}")
    logger.info(f"  Drug indices range: [{sample_batch[0].min()}, {sample_batch[0].max()}]")
    logger.info(f"  Protein indices range: [{sample_batch[1].min()}, {sample_batch[1].max()}]")
    logger.info(f"  Labels unique values: {torch.unique(sample_batch[2])}")

    # Initialize model
    logger.info("\n" + "="*80)
    logger.info("STEP 3: INITIALIZING MODEL")
    logger.info("="*80)

    model = EMMDTI(
        vocab_size=len(data_module.fcs_vocab),
        fcs_embedding_dim=128,
        mamba_hidden_dim=64,  # Standardized to match other 9 architectures
        mamba_n_layers=2,
        mamba_state_size=16,
        cnn_out_channels=3,
        dropout=0.1,
    )

    logger.info(f"✓ Model created")
    logger.info(model)

    # Move model to device
    model = model.to(device)
    logger.info(f"Model moved to device: {device}")

    # Test forward pass with detailed debugging
    logger.info("\nTesting forward pass with DETAILED DEBUGGING...")
    try:
        # Monkey-patch to capture intermediate values
        drug_indices = sample_batch[0].to(device)
        protein_indices = sample_batch[1].to(device)

        with torch.no_grad():
            # Step 1: Embedding
            drug_emb = model.embedding(drug_indices)
            drug_emb = model.embedding_norm(drug_emb)
            logger.info(f"  Drug embeddings: shape={drug_emb.shape}, range=[{drug_emb.min():.4f}, {drug_emb.max():.4f}]")

            protein_emb = model.embedding(protein_indices)
            protein_emb = model.embedding_norm(protein_emb)
            logger.info(f"  Protein embeddings: shape={protein_emb.shape}, range=[{protein_emb.min():.4f}, {protein_emb.max():.4f}]")

            # Step 2: Mamba
            drug_repr = model.drug_mamba(drug_emb)
            logger.info(f"  Drug Mamba output: shape={drug_repr.shape}, range=[{drug_repr.min():.4f}, {drug_repr.max():.4f}]")

            protein_repr = model.protein_mamba(protein_emb)
            logger.info(f"  Protein Mamba output: shape={protein_repr.shape}, range=[{protein_repr.min():.4f}, {protein_repr.max():.4f}]")

            # Step 3: Pooling
            drug_pool = drug_repr.mean(dim=1)
            protein_pool = protein_repr.mean(dim=1)
            logger.info(f"  Drug pool: shape={drug_pool.shape}, range=[{drug_pool.min():.4f}, {drug_pool.max():.4f}]")
            logger.info(f"  Protein pool: shape={protein_pool.shape}, range=[{protein_pool.min():.4f}, {protein_pool.max():.4f}]")

            # Step 4: Interaction matrix
            interaction_matrix = torch.bmm(drug_repr, protein_repr.transpose(1, 2))
            logger.info(f"  Interaction matrix: shape={interaction_matrix.shape}, range=[{interaction_matrix.min():.4f}, {interaction_matrix.max():.4f}]")
            logger.info(f"  Interaction matrix unique values (first 10): {torch.unique(interaction_matrix.flatten())[:10]}")

            interaction_matrix = interaction_matrix.unsqueeze(1)

            # Step 5: CNN
            cnn_features = model.interaction_cnn(interaction_matrix)
            logger.info(f"  CNN features: shape={cnn_features.shape}, range=[{cnn_features.min():.4f}, {cnn_features.max():.4f}]")

            cnn_features_flat = cnn_features.view(drug_indices.size(0), -1)
            logger.info(f"  CNN flattened: shape={cnn_features_flat.shape}, range=[{cnn_features_flat.min():.4f}, {cnn_features_flat.max():.4f}]")

            # Step 6: MLP input
            final_features = torch.cat([cnn_features_flat, drug_pool, protein_pool], dim=1)
            logger.info(f"  MLP input: shape={final_features.shape}, range=[{final_features.min():.4f}, {final_features.max():.4f}]")

            # Step 7: Prediction
            output = model.predictor(final_features)
            logger.info(f"✓ Forward pass successful")
            logger.info(f"  Output shape: {output.shape}")
            logger.info(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")
            logger.info(f"  Output values (first 5): {output[:5].squeeze().tolist()}")
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}", exc_info=True)
        return

    # Train
    logger.info("\n" + "="*80)
    logger.info("STEP 4: TRAINING")
    logger.info("="*80)

    output_dir = log_dir / "checkpoints"
    trainer = Trainer(model, device, output_dir=output_dir)

    try:
        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=10,  # Short training for debugging
            learning_rate=3e-4,
            optimizer_name="adam",
            weight_decay=1e-4,
            gradient_clip=1.0,
            early_stopping_patience=30,
            scheduler_name=None,
        )

        logger.info(f"✓ Training completed")
        logger.info(f"  History keys: {history.keys()}")

    except Exception as e:
        logger.error(f"✗ Training failed: {e}", exc_info=True)
        return

    # Evaluate
    logger.info("\n" + "="*80)
    logger.info("STEP 5: EVALUATION")
    logger.info("="*80)

    try:
        model.load_checkpoint(output_dir / "best_model.pt")

        from emm_dti.training.metrics import Metrics
        import numpy as np

        predictions, targets = trainer.predict(test_loader)
        predictions = np.array(predictions).squeeze()
        targets = np.array(targets)

        logger.info(f"Predictions shape: {predictions.shape}")
        logger.info(f"Predictions range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        logger.info(f"Targets shape: {targets.shape}")
        logger.info(f"Targets unique: {np.unique(targets)}")

        test_metrics = Metrics.compute_metrics(targets, predictions)
        logger.info(f"Test metrics: {Metrics.format_metrics(test_metrics, prefix='test_')}")

    except Exception as e:
        logger.error(f"✗ Evaluation failed: {e}", exc_info=True)
        return

    logger.info("\n" + "="*80)
    logger.info(f"Debug logs saved to: {log_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    debug_training()
