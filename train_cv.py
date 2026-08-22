"""5-Fold CV training for EMM-DTI (FCS + Mamba-SSM + CNN + MLP) architecture."""
import argparse, copy, json, logging, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

logger = None
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from emm_dti.utils.config import Config
from emm_dti.utils.logging_utils import setup_logging
from emm_dti.utils.device import get_device
from emm_dti.data.loaders import DTIDataModule, DTIDataset
from emm_dti.models.emm_dti import EMMDTI
from emm_dti.training.metrics import Metrics

def load_dataset_config(dataset_name: str):
    """Load dataset-specific config from data directory."""
    config_path = PROJECT_ROOT / "configs" / f"train_{dataset_name}.yaml"
    if config_path.exists():
        return Config.from_yaml(str(config_path))
    else:
        # Default config
        return Config.from_yaml(PROJECT_ROOT / "configs" / "default.yaml")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

def run_epoch(model, loader, criterion, optimizer=None, device=DEVICE):
    """Run one epoch of training or evaluation."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, all_labels, all_probs = 0.0, [], []

    with (torch.enable_grad() if is_train else torch.no_grad()):
        for batch in loader:
            # Unpack batch
            if isinstance(batch, (tuple, list)):
                drug_indices, protein_indices, labels = batch
            else:
                drug_indices = batch['drug_indices']
                protein_indices = batch['protein_indices']
                labels = batch['label']

            drug_indices = drug_indices.to(device)
            protein_indices = protein_indices.to(device)
            labels = labels.to(device).float()

            # Forward pass
            logits = model(drug_indices, protein_indices)
            if logits.dim() > 1:
                logits = logits.squeeze(-1)

            loss = criterion(logits, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError("NaN/Inf loss detected")

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    metrics = Metrics.compute_metrics(np.array(all_labels), np.array(all_probs))
    return total_loss / len(loader.dataset), metrics

def train_fold(fold, train_indices, val_indices, dataset, config, checkpoint_dir, device=DEVICE):
    """Train and validate a single fold."""
    # Create model
    model = EMMDTI(
        vocab_size=len(dataset.fcs_vocab),
        fcs_embedding_dim=config.model.fcs_embedding_dim,
        mamba_hidden_dim=config.model.mamba_hidden_dim,
        mamba_n_layers=config.model.mamba_n_layers,
        cnn_out_channels=config.model.cnn_out_channels,
        dropout=config.model.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_pr_auc, best_val_metrics, best_val_loss, wait = -1.0, None, None, 0

    for epoch in range(1, config.training.epochs + 1):
        # Create fold-specific subsets
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        # Create loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=0,  # Avoid issues with FCS patterns
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        train_loss, train_m = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_m = run_epoch(model, val_loader, criterion, device=device)

        # ─── Per-epoch table ───────────────────────────────────────────────
        sep = "    " + "─" * 50
        print(sep)
        print(f"    Fold {fold} | Epoch {epoch}")
        print(sep)

        header = f"    {'Metric':<16}  {'Train':>12}  {'Val':>12}"
        header_sep = f"    {'─'*16}  {'─'*12}  {'─'*12}"
        print(header_sep)
        print(header)
        print(header_sep)

        # Print all metrics
        metric_keys = ["accuracy", "precision", "recall", "specificity", "mcc", "roc_auc", "pr_auc"]
        for key in metric_keys:
            label = key.replace("_", " ").title()
            train_val = train_m.get(key, 0.0)
            val_val = val_m.get(key, 0.0)
            line = f"    {label:<16}  {train_val:>12.4f}  {val_val:>12.4f}"
            print(line)

        # Print loss
        loss_line = f"    {'Loss':<16}  {train_loss:>12.4f}  {val_loss:>12.4f}"
        print(loss_line)
        print(sep)

        if val_m.get("pr_auc", -1) > best_val_pr_auc:
            best_val_pr_auc = val_m.get("pr_auc", -1)
            best_val_metrics = copy.deepcopy(val_m)
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / f"best_model_fold_{fold}.pt")
            wait = 0
        else:
            wait += 1
            if wait >= config.training.early_stopping_patience:
                break

    # Load best model
    if (checkpoint_dir / f"best_model_fold_{fold}.pt").exists():
        model.load_state_dict(torch.load(checkpoint_dir / f"best_model_fold_{fold}.pt", map_location=device))

    return {f"val_{k}": v for k, v in (best_val_metrics or {}).items()} | ({"val_loss": best_val_loss} if best_val_loss else {})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="human", help="Dataset name")
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = Config.from_yaml(PROJECT_ROOT / "configs" / f"train_{args.dataset}.yaml")

    # Override with CLI args
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.training.early_stopping_patience = 30

    # Setup directories
    results_dir = PROJECT_ROOT / "results" / args.dataset
    checkpoint_dir = PROJECT_ROOT / "checkpoints" / args.dataset
    logs_dir = PROJECT_ROOT / "logs" / args.dataset

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    global logger
    log_file = logs_dir / f"{args.dataset}_cv_training.log"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info(f"Starting 5-fold CV training on {args.dataset}")
    logger.info(f"Results dir: {results_dir}")
    logger.info(f"Logs dir: {logs_dir}")
    logger.info(f"Checkpoint dir: {checkpoint_dir}\n")

    # Load data module
    data_module = DTIDataModule(
        config.dataset.data_dir,
        train_split=config.dataset.train_split,
        val_split=config.dataset.val_split,
        test_split=config.dataset.test_split,
        random_seed=config.dataset.random_seed,
    )

    # Create dataset
    drug_sequences = [data_module.drugs[drug_id] for drug_id in data_module.interactions_df["drug_id"]]
    protein_sequences = [data_module.proteins[protein_id] for protein_id in data_module.interactions_df["protein_id"]]
    interactions_list = data_module.interactions_df["interaction"].tolist()

    dataset = DTIDataset(
        drug_sequences=drug_sequences,
        protein_sequences=protein_sequences,
        interactions=interactions_list,
        fcs_vocab=data_module.fcs_vocab,
        fcs_patterns=data_module.fcs.get_patterns(),
    )

    interactions = data_module.interactions_df.copy()

    # CV with multiple seeds
    CV_SEEDS = [42, 123, 2024]
    all_results = []

    for seed_idx, cv_seed in enumerate(CV_SEEDS, 1):
        print(f"\n{'='*70}\n  CV Run {seed_idx}/3 (seed={cv_seed})\n{'='*70}")
        np.random.seed(cv_seed)
        torch.manual_seed(cv_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cv_seed)

        fold_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(
            StratifiedKFold(n_splits=5, shuffle=True, random_state=cv_seed).split(
                interactions, interactions["interaction"]
            ),
            1
        ):
            train_df = interactions.iloc[train_idx].reset_index(drop=True)
            val_df = interactions.iloc[val_idx].reset_index(drop=True)
            print(f"  Fold {fold_idx}/5 | Train: {len(train_df):,} | Val: {len(val_df):,}")

            fold_metrics = train_fold(fold_idx, train_idx, val_idx, dataset, config, checkpoint_dir, DEVICE)
            fold_results.append(fold_metrics)

        all_results.append(fold_results)

    # Aggregate results
    print(f"\n{'='*70}\n  SUMMARY: 3 CV Runs × 5 Folds\n{'='*70}")
    results_data = []
    for seed_idx, cv_seed in enumerate(CV_SEEDS, 1):
        cv_fold_results = all_results[seed_idx - 1]
        for fold_idx, fold_metrics in enumerate(cv_fold_results, 1):
            row = {"seed": cv_seed, "fold": fold_idx}
            for key, val in fold_metrics.items():
                row[key] = val
            results_data.append(row)

    # Summary statistics
    summary_metrics = {}
    metric_keys = ["val_pr_auc", "val_roc_auc", "val_accuracy", "val_precision", "val_recall", "val_specificity", "val_mcc"]
    for metric_key in metric_keys:
        all_vals = [row[metric_key] for row in results_data if metric_key in row]
        if all_vals:
            summary_metrics[metric_key] = {"mean": float(np.mean(all_vals)), "std": float(np.std(all_vals))}
            print(f"  {metric_key:<20}: {np.mean(all_vals):.4f} ± {np.std(all_vals):.4f}")

    # Save results
    results_csv_path = results_dir / "results.csv"
    pd.DataFrame(results_data).to_csv(results_csv_path, index=False)
    print(f"\n  ✓ Saved fold results to {results_csv_path}")

    summary_data = {
        "dataset": args.dataset,
        "num_seeds": len(CV_SEEDS),
        "num_folds": 5,
        "total_folds": len(results_data),
        "metrics": summary_metrics
    }
    results_json_path = results_dir / "results.json"
    with open(results_json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  ✓ Saved summary to {results_json_path}")

    cv_summary_file = results_dir / "cv_summary.json"
    with open(cv_summary_file, "w") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"  ✓ Saved CV summary to {cv_summary_file}")

if __name__ == "__main__":
    main()
