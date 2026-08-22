# EMM-DTI 5-Fold Cross-Validation Training

This script implements stratified 5-fold cross-validation with 3 independent CV runs for the EMM-DTI model, maintaining strict separation of the training setup.

## Architecture

**EMM-DTI Pipeline:**
```
Drug SMILES → FCS Mining → Fragment Encoding → Bidirectional Mamba-SSM → Interaction Matrix (Dot Product) → CNN Feature Extraction → MLP Predictor → DTI Logit
```

## Configuration

Default settings (from `configs/train_human_ssm.yaml`):
- **Epochs:** 200
- **Early Stopping Patience:** 30
- **Batch Size:** 16
- **Learning Rate:** 3e-4 (0.0003)
- **Optimizer:** Adam (weight_decay=1e-4)
- **Gradient Clipping:** 1.0

Model Architecture:
- **FCS Embedding Dim:** 128
- **Mamba Hidden Dim:** 256
- **Mamba Layers:** 2
- **Mamba State Size:** 16
- **CNN Output Channels:** 3
- **Dropout:** 0.1

## Usage

### Basic Training
```bash
python train_cv.py --dataset human --config configs/train_human_ssm.yaml
```

### With Custom Hyperparameters
```bash
python train_cv.py \
  --dataset human \
  --epochs 200 \
  --batch_size 16 \
  --lr 3e-4 \
  --config configs/train_human_ssm.yaml
```

## Output Structure

Training results are organized as:
```
results/
├── human/
│   ├── results.csv        # Individual fold results (15 total: 3 seeds × 5 folds)
│   ├── results.json       # Summary statistics with mean ± std
│   └── cv_summary.json    # CV metrics summary

checkpoints/
├── human/
│   ├── best_model_fold_1.pt
│   ├── best_model_fold_2.pt
│   └── ... (5 folds per CV run)

logs/
└── human/
    └── cv_training.log    # Detailed training log
```

## Cross-Validation Setup

### CV Runs
- **Number of Runs:** 3
- **Seeds:** [42, 123, 2024]
- **Folds per Run:** 5 (Stratified K-Fold)
- **Total Folds:** 15

### Split Strategy
Each CV run performs stratified 5-fold cross-validation:
1. Stratifies by interaction labels to maintain class distribution
2. FCS patterns mined from **training data only** (no data leakage)
3. Separate train/val splits for each fold

## Metrics Tracked

- **Accuracy**
- **Precision**
- **Recall**
- **Specificity**
- **Matthews Correlation Coefficient (MCC)**
- **ROC-AUC**
- **PR-AUC** (used for early stopping)

## Key Implementation Details

### Data Loading
- `DTIDataModule` loads drugs.csv, proteins.csv, and interactions.csv
- FCS patterns mined from training sequences only
- Fragment vocabulary built from mined patterns

### Model
- `EMMDTI` class with bidirectional Mamba-SSM encoder
- Interaction matrix computed via dot product (drug_repr @ protein_repr^T)
- CNN extracts features from interaction matrix
- MLP decoder (3 layers) predicts binary DTI logit

### Training Loop
- Early stopping based on validation PR-AUC
- Patience: 30 epochs
- Checkpoint saved only on best validation PR-AUC
- Gradient clipping: 1.0
- Mixed precision: disabled (standard FP32)

## Reproducibility

- Random seeds set for NumPy, PyTorch, and CUDA
- FCS patterns cached to disk for reproducibility
- Stratified K-Fold ensures balanced class distribution per fold
- All 15 fold results and summary statistics saved automatically

## Typical Output

```
=================================================================================
  CV Run 1/3 (seed=42)
=================================================================================
  Fold 1/5 | Train: 6,104 | Val: 1,368

    ──────────────────────────────────────────────────
    Fold 1 | Epoch 1
    ──────────────────────────────────────────────────
    ────────────────  ────────────────  ────────────────
    Metric               Train          Val
    ────────────────  ────────────────  ────────────────
    Accuracy             0.7234          0.7156
    Precision            0.7512          0.7402
    Recall               0.6891          0.6824
    Specificity          0.7578          0.7489
    Mcc                  0.4521          0.4312
    Roc Auc              0.8234          0.8156
    Pr Auc               0.8156          0.8042
    Loss                 0.5432          0.5678
    ──────────────────────────────────────────────────

...

=================================================================================
  SUMMARY: 3 CV Runs × 5 Folds
=================================================================================
  val_pr_auc          : 0.8342 ± 0.0124
  val_roc_auc         : 0.8567 ± 0.0156
  val_accuracy        : 0.8234 ± 0.0098
  val_precision       : 0.8156 ± 0.0134
  val_recall          : 0.8267 ± 0.0156
  val_specificity     : 0.8201 ± 0.0145
  val_mcc             : 0.6478 ± 0.0187
```

## Requirements

- PyTorch with CUDA support (for GPU training)
- scikit-learn (for StratifiedKFold)
- pandas, numpy
- mamba-ssm library
- EMM-DTI model and utilities

See `setup.py` and `requirements.txt` for full dependencies.
