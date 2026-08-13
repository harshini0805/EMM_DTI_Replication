# EMM-DTI: Enhanced Mamba-Based Model for Drug-Target Interaction Prediction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A PyTorch implementation of EMM-DTI for predicting drug-target interactions using Mamba-based sequence models with Frequent Continuous Subsequence (FCS) mining.

## Overview

EMM-DTI predicts whether a drug molecule will interact with a target protein by:
1. **FCS Module**: Mining frequent substructures from SMILES and amino acid sequences
2. **Mamba Layers**: Capturing bidirectional long-range dependencies with linear-time complexity
3. **Interaction Matrix**: Computing 2D feature maps via dot products
4. **CNN Predictor**: Binary classification via Conv2D + MLP

### Key Results (Human Dataset)
- **AUC**: 0.993 | **AUPR**: 0.983 | **Accuracy**: 0.957
- **Precision**: 0.972 | **Recall**: 0.962

## Quick Start

### Installation

```bash
# Clone repository
git clone <repo-url>
cd emm-dti

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train on Human dataset with default config
python -m emm_dti.train --config configs/train_human.yaml

# Override hyperparameters
python -m emm_dti.train \
    --config configs/train_human.yaml \
    --batch_size 64 \
    --learning_rate 0.001 \
    --epochs 100
```

### Evaluation

```bash
# Evaluate on test set
python -m emm_dti.evaluate \
    --checkpoint results/human/best_model.pt \
    --data_dir data/human
```

## Project Structure

```
emm-dti/
├── emm_dti/                    # Source code
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mamba.py           # Mamba layer implementation
│   │   ├── fcs.py             # FCS mining module
│   │   └── emm_dti.py         # Full EMM-DTI model
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py         # DataLoader implementations
│   │   └── preprocessing.py   # Data preprocessing
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py         # Training loop
│   │   └── metrics.py         # Evaluation metrics
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── config.py          # Configuration management
│   │   ├── logging.py         # Logging utilities
│   │   └── device.py          # Device management (GPU/CPU)
│   ├── train.py               # Training entry point
│   └── evaluate.py            # Evaluation entry point
├── configs/                    # Configuration files
│   ├── train_human.yaml       # Human dataset config
│   ├── train_biosnap.yaml     # BIOSNAP dataset config
│   └── default.yaml           # Default hyperparameters
├── data/                       # Dataset directory
│   ├── human/
│   │   ├── drugs.csv
│   │   ├── proteins.csv
│   │   └── interactions.csv
│   └── biosnap/
├── results/                    # Training outputs
│   ├── checkpoints/
│   ├── logs/
│   └── metrics/
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_data.py
│   └── test_metrics.py
├── notebooks/                  # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_model_analysis.ipynb
│   └── 03_results.ipynb
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── .gitignore
├── .env.example               # Environment variables template
└── README.md
```

## Configuration

All hyperparameters are managed through YAML config files in `configs/`:

```yaml
# configs/train_human.yaml
dataset:
  name: human
  data_dir: data/human
  train_split: 0.7
  val_split: 0.2
  test_split: 0.1

model:
  fcs_embedding_dim: 128
  mamba_hidden_dim: 256
  mamba_n_layers: 2
  dropout: 0.1
  
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 100
  early_stopping_patience: 10
  device: cuda
  
optimization:
  optimizer: adam
  weight_decay: 1e-5
  gradient_clip: 1.0
```

## Data Format

### Drugs (SMILES)
```csv
drug_id,smiles,name
D001,CC(=O)Oc1ccccc1C(=O)O,Aspirin
D002,CN1C(=O)CC(c2ccccc2)C1=O,Phenytoin
```

### Proteins (Sequences)
```csv
protein_id,sequence,gene_name
P001,MSVPTSSMFFHQSN...,TP53
P002,MEYFTVGYPPN...,EGFR
```

### Interactions
```csv
drug_id,protein_id,interaction
D001,P001,1
D001,P002,0
```

## Model Architecture

### FCS Mining Module
- Identifies frequent k-mers in SMILES/sequences using Apriori algorithm
- Extracts diverse fragment types: branch chains, common substructures, motifs
- Embedding + Layer Normalization for training stability

### Mamba Layers (Selective State Space Model)
```
h_t = A_t * h_{t-1} + B_t * x_t
y_t = C_t^T * h_t
```
- Bidirectional processing of drug and protein sequences
- Linear-time complexity O(N) vs quadratic attention O(N²)
- Captures long-range dependencies effectively

### Prediction Head
1. **Interaction Matrix**: Dot product of Mamba outputs → (batch, D, P)
2. **CNN Layer**: 2D convolution (3×3 kernel) → 3-channel feature map
3. **MLP**: Fully connected layers → binary classification

## Metrics

All metrics computed on test set with 5-fold cross-validation:

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **AUC** | Area under ROC curve | Overall classification performance |
| **AUPR** | Area under Precision-Recall | Performance on positive class |
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| **Precision** | TP/(TP+FP) | False positive rate |
| **Recall** | TP/(TP+FN) | False negative rate |

## Training

### Hardware Requirements
- GPU: NVIDIA RTX3090 or equivalent (8GB+ VRAM)
- RAM: 32GB recommended
- Storage: 20GB for dataset + checkpoints

### Training Command
```bash
# With tensorboard logging
python -m emm_dti.train \
    --config configs/train_human.yaml \
    --log_dir results/human \
    --tensorboard
```

### Monitoring
```bash
# View training progress
tensorboard --logdir results/human/logs
```

## Evaluation & Inference

### Test Set Evaluation
```bash
python -m emm_dti.evaluate \
    --checkpoint results/human/best_model.pt \
    --data_dir data/human \
    --output_dir results/human/evaluation
```

### Predict New Interactions
```python
from emm_dti.models import EMMDTI
from emm_dti.data import preprocess_drug, preprocess_protein

model = EMMDTI.load('results/human/best_model.pt')
drug_smiles = "CC(=O)Oc1ccccc1C(=O)O"
protein_seq = "MSVPTSSMFFHQSN..."

pred = model.predict(drug_smiles, protein_seq)
print(f"Interaction probability: {pred:.3f}")
```

## References

### Primary Paper
- Sun et al. (2025). "EMM-DTI: Enhanced Mamba-Based Model for Drug-Target Interaction Prediction"

### Related Work
- Dou et al. (2023). "BCM-DTI: A fragment-oriented method for drug-target interaction prediction using deep learning"
- Hu & Chan (2015). "Discovering Variable-Length Patterns in Protein Sequences for Protein-Protein Interaction Prediction"

## Contributing

We follow professional ML development practices:

1. **Code Style**: PEP 8 via `black` and `flake8`
2. **Type Hints**: Full type annotations required
3. **Testing**: Unit tests in `tests/` with `pytest`
4. **Documentation**: Docstrings following Google style guide
5. **Git**: Feature branches, meaningful commits, code review

### Development Setup
```bash
pip install -r requirements-dev.txt
pre-commit install
```

### Running Tests
```bash
pytest tests/ -v --cov=emm_dti
```

## License

MIT License - see LICENSE file

## Citation

```bibtex
@article{sun2025emmdti,
  title={EMM-DTI: Enhanced Mamba-Based Model for Drug-Target Interaction Prediction},
  author={Sun, Qinglong and Zhang, Jun and Hu, Pengwei and Qi, Xiangwei and Hu, Lun},
  booktitle={EITCE 2025},
  year={2025}
}
```

## Contact

For issues, questions, or contributions, please open an issue or contact the development team.

---

**Last Updated**: July 24, 2026 | **Status**: Active Development
