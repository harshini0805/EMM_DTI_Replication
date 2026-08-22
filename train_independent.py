"""
Run 5 independent training runs for a dataset using the standard train/val/test splits.
Aggregates and reports the metrics across all 5 runs.
"""
import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from emm_dti.utils.config import Config
from emm_dti.utils.device import get_device
from emm_dti.data.loaders import DTIDataModule
from emm_dti.models.emm_dti import EMMDTI
from emm_dti.training.trainer import Trainer
from emm_dti.training.metrics import Metrics

def main():
    parser = argparse.ArgumentParser(description="5 Independent Runs for EMM-DTI")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    args = parser.parse_args()

    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config_path = PROJECT_ROOT / "configs" / f"train_{args.dataset}.yaml"
        if not config_path.exists():
            config_path = PROJECT_ROOT / "configs" / "default.yaml"
        config = Config.from_yaml(str(config_path))

    # Seeds for the 5 independent runs
    INDEPENDENT_SEEDS = [42, 123, 1024, 2024, 9999]
    
    results_dir = PROJECT_ROOT / "results" / f"{args.dataset}_independent"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = results_dir / f"{args.dataset}_independent_runs.log"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    
    # Console output
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console)

    logger.info(f"Starting 5 Independent Runs on {args.dataset}")
    logger.info(f"Results dir: {results_dir}\n")

    all_test_metrics = []

    for run_idx, seed in enumerate(INDEPENDENT_SEEDS, 1):
        logger.info(f"\n{'='*70}\n  Run {run_idx}/5 (seed={seed})\n{'='*70}")
        
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            
        config.dataset.random_seed = seed
        
        device = get_device(config.training.device)
        
        # Load Data
        data_module = DTIDataModule(
            data_dir=config.dataset.data_dir,
            train_split=config.dataset.train_split,
            val_split=config.dataset.val_split,
            test_split=config.dataset.test_split,
            random_seed=config.dataset.random_seed,
        )
        
        train_loader, val_loader, test_loader = data_module.create_loaders(
            batch_size=config.training.batch_size,
            num_workers=config.training.num_workers,
            pin_memory=config.training.pin_memory,
        )
        
        # Initialize Model
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
        
        run_output_dir = results_dir / f"run_{run_idx}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        
        trainer = Trainer(model, device, output_dir=run_output_dir)
        
        # Train
        try:
            trainer.fit(
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
            
            # Evaluate on Test Set
            model.load_checkpoint(run_output_dir / "best_model.pt")
            
            predictions, targets = trainer.predict(test_loader)
            predictions = np.array(predictions).squeeze()
            targets = np.array(targets)
            
            test_metrics = Metrics.compute_metrics(targets, predictions)
            test_metrics['seed'] = seed
            test_metrics['run'] = run_idx
            
            logger.info(f"\nRun {run_idx} Test Metrics:")
            for k, v in test_metrics.items():
                if isinstance(v, float):
                    logger.info(f"  {k}: {v:.4f}")
                    
            all_test_metrics.append(test_metrics)
            
        except Exception as e:
            logger.error(f"Run {run_idx} failed: {e}")

    # Summary
    logger.info(f"\n{'='*70}\n  SUMMARY: 5 Independent Runs\n{'='*70}")
    
    summary_metrics = {}
    metric_keys = ["accuracy", "precision", "recall", "specificity", "mcc", "roc_auc", "pr_auc"]
    for metric_key in metric_keys:
        vals = [run[metric_key] for run in all_test_metrics if metric_key in run]
        if vals:
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            summary_metrics[metric_key] = {"mean": mean_val, "std": std_val}
            logger.info(f"  {metric_key:<16}: {mean_val:.4f} ± {std_val:.4f}")

    # Save Results
    pd.DataFrame(all_test_metrics).to_csv(results_dir / "all_runs_results.csv", index=False)
    
    with open(results_dir / "summary_results.json", "w") as f:
        json.dump(summary_metrics, f, indent=2)

    logger.info(f"\n  ✓ Saved all results to {results_dir}")

if __name__ == "__main__":
    main()
