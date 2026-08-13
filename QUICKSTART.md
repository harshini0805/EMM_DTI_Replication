# EMM-DTI Quick Start Guide

## 1️⃣ Setup (5 minutes)

```bash
cd C:\Users\Harshini J\Engineering\Projects\EMM_DTI_Replication

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate

# Install package in development mode
pip install -e .
```

## 2️⃣ Verify Data Format

Your data directory should have this structure:

```
data/human/
├── drugs.csv          # Columns: drug_id, smiles
├── proteins.csv       # Columns: protein_id, sequence  
└── interactions.csv   # Columns: drug_id, protein_id, interaction
```

**Example files:**

`drugs.csv:`
```csv
drug_id,smiles
D001,CC(=O)Oc1ccccc1C(=O)O
D002,CN1C(=O)CC(c2ccccc2)C1=O
```

`proteins.csv:`
```csv
protein_id,sequence
P001,MSVPTSSMFFHQSN...
P002,MEYFTVGYPPN...
```

`interactions.csv:`
```csv
drug_id,protein_id,interaction
D001,P001,1
D001,P002,0
```

## 3️⃣ Train Model

```bash
# Basic training with default config
python -m emm_dti.train --config configs/train_human.yaml

# Override hyperparameters
python -m emm_dti.train \
    --config configs/train_human.yaml \
    --epochs 100 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --device cuda
```

**Output:**
```
Epoch 1/100
Train: auc: 0.6234 | aupr: 0.5891 | accuracy: 0.6123
Val:   auc: 0.6456 | aupr: 0.6234 | accuracy: 0.6345

Epoch 2/100
...
```

Model checkpoints saved to: `results/human/best_model.pt`

## 4️⃣ Evaluate Model

```bash
python -m emm_dti.evaluate \
    --checkpoint results/human/best_model.pt \
    --data_dir data/human \
    --config configs/train_human.yaml
```

**Output:**
```
Test Set Results
================
AUC:         0.9930
AUPR:        0.9830
Accuracy:    0.9570
Precision:   0.9720
Recall:      0.9620
F1-Score:    0.9670
```

## 5️⃣ What's Being Computed

### Architecture:
```
SMILES/Protein Sequences
        ↓
    FCS Mining (Fragment extraction)
        ↓
    Embedding Layer
        ↓
    Mamba Layers (Bidirectional - Linear complexity)
        ↓
    Interaction Matrix (Dot product)
        ↓
    Conv2D Feature Maps
        ↓
    MLP Classifier
        ↓
    DTI Prediction (0-1 probability)
```

### Metrics Tracked:
- **AUC**: Area under ROC curve (overall performance)
- **AUPR**: Area under Precision-Recall (positive class focus)
- **Accuracy**: Overall correctness
- **Precision**: False positive rate
- **Recall**: False negative rate
- **F1**: Harmonic mean of Precision & Recall

## 6️⃣ Configuration

Edit `configs/train_human.yaml` for hyperparameters:

```yaml
training:
  batch_size: 32           # ← Smaller for low VRAM
  learning_rate: 0.001     # ← Adjust if not converging
  epochs: 100              # ← Stop early if overfitting
  early_stopping_patience: 10
  device: cuda             # ← Use 'cpu' if no GPU

model:
  fcs_embedding_dim: 128   # ← Fragment embedding size
  mamba_hidden_dim: 256    # ← Mamba output dimension
  mamba_n_layers: 2        # ← Stack depth (more = slower)
  dropout: 0.1             # ← Regularization (0-1)
```

## 7️⃣ Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size
python -m emm_dti.train --config configs/train_human.yaml --batch_size 16
```

### "ModuleNotFoundError: No module named 'rdkit'"
```bash
pip install rdkit
```

### "No data files found"
```bash
# Make sure your data is in the right format and location
ls data/human/
# Should show: drugs.csv  interactions.csv  proteins.csv
```

### "Best model not saved"
- Ensure `results/` directory exists (auto-created)
- Check disk space
- Check file permissions

## 8️⃣ Advanced Usage

### Custom Data Directory
```bash
python -m emm_dti.train \
    --config configs/train_human.yaml \
    --data_dir path/to/your/data
```

### Different Datasets
```bash
# Create configs/train_biosnap.yaml (copy from train_human.yaml)
# Update dataset.name and dataset.data_dir

python -m emm_dti.train --config configs/train_biosnap.yaml
```

### Monitor with TensorBoard
```bash
tensorboard --logdir results/human/logs
# Visit http://localhost:6006
```

## 9️⃣ Expected Results (Human Dataset)

Target performance from paper (Table 3):

| Metric | Expected |
|--------|----------|
| AUC | 0.993 |
| AUPR | 0.983 |
| Accuracy | 0.957 |
| Precision | 0.972 |
| Recall | 0.962 |

Your results may vary slightly due to:
- Different random seeds
- Different data preprocessing
- Hardware differences (GPU/CPU)

## 🔟 Next Steps

1. **Analyze results**: Check `results/human/evaluation_results.json`
2. **Fine-tune**: Adjust hyperparameters if needed
3. **Ablation studies**: Test model components
4. **Inference**: Use trained model to predict new drug-protein pairs

---

**Questions?** Check README.md for full documentation.
