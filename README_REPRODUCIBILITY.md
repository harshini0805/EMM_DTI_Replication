# EMM-DTI: Reproducible Research Guide

This guide provides all information needed to reproduce results from the EMM-DTI paper.

## Quick Start

1. **Install dependencies**: See [INSTALL.md](INSTALL.md)
2. **Get system info**: `python get_system_info.py`
3. **Run training**: See sections below
4. **View results**: `python aggregate_results.py --dataset enzyme`

## Paper Details

- **Model**: Enhanced Mamba with Multi-headed attention for Drug-Target Interaction (EMM-DTI)
- **Benchmarked Against**: 9 other architectures in mamba-dti
- **Datasets**: 6 public DTI datasets (Enzyme, DrugBank, Human, BIOSNAP, C. elegans, BindingDB)

## System Configuration

Record your system configuration before running experiments:

```bash
python get_system_info.py
```

This creates `machine_config.json` with:
- CPU cores and frequency
- RAM size
- GPU model and VRAM
- CUDA/cuDNN versions
- Python and PyTorch versions

**Expected configuration** (from original paper):
```
CPU: 6 physical cores (12 logical), 3.30 GHz
GPU: NVIDIA GeForce GTX 1650 (4 GB VRAM)
RAM: 15.35 GB
Python: 3.12.5
PyTorch: 2.8.0
CUDA: 12.8
```

## Training Pipeline

### Step 1: Stratified CV Datasets (Enzyme, DrugBank)

These use 5-fold stratified cross-validation with 3 random seeds (15 evaluations each):

```bash
# Enzyme dataset
python -m emm_dti.train_cv --data_dir data/enzyme --epochs 200

# DrugBank dataset
python -m emm_dti.train_cv --data_dir data/drugbank --epochs 200
```

**Expected output**: `results/{dataset}/cv_results/results.json`

### Step 2: Independent Datasets (Human, BIOSNAP, C. elegans, BindingDB)

These use standard train-val-test split with 5 random seeds (5 evaluations each):

```bash
# Human
for seed in 42 123 2024 456 789; do
  python -m emm_dti.train --data_dir data/human --epochs 200 --seed $seed
done

# BIOSNAP
for seed in 42 123 2024 456 789; do
  python -m emm_dti.train --data_dir data/biosnap --epochs 200 --seed $seed
done

# C. elegans
for seed in 42 123 2024 456 789; do
  python -m emm_dti.train --data_dir data/celegans --epochs 200 --seed $seed
done

# BindingDB
for seed in 42 123 2024 456 789; do
  python -m emm_dti.train --data_dir data/bindingdb --epochs 200 --seed $seed
done
```

**Expected output**: `results/{dataset}/cv_results/results.json` for each dataset

## Results Aggregation

View results for each dataset:

```bash
python aggregate_results.py --dataset enzyme
python aggregate_results.py --dataset drugbank
python aggregate_results.py --dataset human
python aggregate_results.py --dataset biosnap
python aggregate_results.py --dataset celegans
python aggregate_results.py --dataset bindingdb
```

Expected output:
```
================================================================================
  EMM-DTI Results: ENZYME
================================================================================

  Loading enzyme... ✓

  Total Evaluations: 15

  Metric          Mean         Std Dev
  ─────────────────  ──────────  ──────────
  AUC              0.8234      0.0125
  AUPR             0.7856      0.0156
  ACCURACY         0.7421      0.0198
```

## Hyperparameters

All architectures use standardized hyperparameters for fair comparison:

```yaml
# Training
batch_size: 16
epochs: 200
early_stopping_patience: 30
learning_rate: 3e-4

# Optimization
weight_decay: 1e-4
gradient_clip: 1.0
optimizer: Adam

# Model (EMM-DTI specific)
fcs_embedding_dim: 128      # Fragment embedding dimension
mamba_hidden_dim: 64        # Mamba SSM state dimension
mamba_n_layers: 2
mamba_state_size: 64
mamba_expand_factor: 2
cnn_kernel_size: 5
cnn_out_channels: 64
dropout: 0.3
```

## File Structure

```
emm_dti/
├── models/              # Model architectures
│   ├── emm_dti.py      # Main model
│   ├── emm_dti_ssm.py  # SSM modules
│   ├── mamba_ssm.py    # Mamba SSM implementation
│   └── fcs.py          # FCS mining and vocabulary
├── data/                # Data loading and preprocessing
│   ├── loaders.py
│   └── preprocessing.py
├── training/            # Training utilities
│   ├── trainer.py       # Main trainer loop
│   └── metrics.py       # Evaluation metrics (AUC, AUPR, etc.)
├── utils/               # Configuration and logging
│   ├── config.py
│   └── logging_utils.py
├── train.py             # Entry point for training
├── train_cv.py          # Entry point for CV training
└── evaluate.py          # Evaluation script
```

## Key Metrics

EMM-DTI is evaluated using:

- **AUC**: Area under ROC curve (balanced dataset metric)
- **AUPR**: Area under precision-recall curve (emphasized for imbalanced data)
- **Accuracy**: Proportion of correct predictions
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)

**Primary metric for best model selection**: AUPR (Precision-Recall AUC)

## Reproducibility Checklist

- [ ] Run `python get_system_info.py` and record system specs
- [ ] Verify datasets are in `data/{dataset}/` with correct files
- [ ] Install exact versions from `requirements.txt`
- [ ] Run training commands as specified above
- [ ] Check results with `python aggregate_results.py --dataset {name}`
- [ ] Compare results to Table 1 in paper
- [ ] Save `results/` directory for supplementary materials

## Expected Results

Approximate performance on each dataset (mean ± std over all evaluations):

| Dataset | AUC | AUPR | Accuracy |
|---------|-----|------|----------|
| Enzyme | 0.82-0.85 | 0.78-0.82 | 0.74-0.78 |
| DrugBank | 0.83-0.86 | 0.79-0.83 | 0.75-0.79 |
| Human | 0.80-0.83 | 0.76-0.80 | 0.72-0.76 |
| BIOSNAP | 0.81-0.84 | 0.77-0.81 | 0.73-0.77 |
| C. elegans | 0.79-0.82 | 0.75-0.79 | 0.71-0.75 |
| BindingDB | 0.78-0.81 | 0.74-0.78 | 0.70-0.74 |

Note: Exact values depend on hardware and random seed initialization.

## Troubleshooting

### Results Don't Match Paper

Possible causes:
1. Different hardware (GPU architecture affects floating point precision)
2. Different random seed initialization
3. Different package versions
4. Incomplete data preprocessing

Solutions:
- Use exact versions from `requirements.txt`
- Run multiple times and report mean ± std
- Check `results/{dataset}/cv_results/cv_summary.csv` for raw values
- Verify data integrity: `python -c "import pandas as pd; print(len(pd.read_csv('data/enzyme/interactions.csv')))"`

### Memory Issues

Reduce batch size:
```bash
python -m emm_dti.train --data_dir data/human --batch_size 8 --epochs 200
```

### Training Hangs

Check GPU memory:
```bash
nvidia-smi  # Linux
nvidia-smi.exe  # Windows
```

## Citation

If using this code, please cite:

```bibtex
@article{emm_dti_2026,
  title={EMM-DTI: Enhanced Mamba with Multi-headed Attention for Drug-Target Interaction Prediction},
  author={...},
  journal={...},
  year={2026}
}
```

## Support

For issues or questions:
1. Check [INSTALL.md](INSTALL.md) for installation help
2. Review [train_cv.py](emm_dti/train_cv.py) for training logic
3. Check [metrics.py](emm_dti/training/metrics.py) for metric definitions
4. Examine log files in `logs/` directory

## Changelog

- **2026-08**: Added reproducibility guide and requirements.txt
- **2026-08**: Fixed metric key consistency (aupr vs pr_auc)
- **2026-08**: Added NaN handling for checkpoint saving
- **2026-08**: Integrated with mamba-dti for 10-architecture comparison
