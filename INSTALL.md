# EMM-DTI Installation Guide

This document provides step-by-step instructions for installing EMM-DTI for reproducible research.

## System Requirements

- **Python**: 3.10 or higher (tested with 3.12.5)
- **CUDA**: 12.1+ (optional, for GPU acceleration)
- **RAM**: Minimum 8 GB (16+ GB recommended)
- **GPU**: NVIDIA GPU with CUDA compute capability 7.0+ (tested with GTX 1650)

See `machine_config.json` for hardware specifications used in this work.

## Installation Steps

### 1. Clone or Set Up the Repository

```bash
cd D:\Projects\EMM_DTI_Replication
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
venv\Scripts\activate

# Or using conda
conda create -n emm_dti python=3.12
conda activate emm_dti
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import numpy; print(f'NumPy {numpy.__version__}')"
python -c "import sklearn; print(f'Scikit-learn {sklearn.__version__}')"
```

### 5. (Optional) Install RDKit for SMILES Canonicalization

```bash
# Via pip (conda recommended for stability)
pip install rdkit

# Or via conda (preferred)
conda install -c conda-forge rdkit
```

## Dataset Preparation

Place your datasets in `data/` directory with structure:

```
data/
├── enzyme/
│   ├── drugs.csv       # Columns: drug_id, smiles
│   ├── proteins.csv    # Columns: protein_id, sequence
│   └── interactions.csv # Columns: drug_id, protein_id, interaction
├── drugbank/
│   ├── drugs.csv
│   ├── proteins.csv
│   └── interactions.csv
├── human/
├── biosnap/
├── celegans/
└── bindingdb/
```

## Quick Start

### Training on CV Dataset (enzyme)

```bash
python -m emm_dti.train_cv --data_dir data/enzyme --epochs 200
```

### Training on Independent Dataset (human) with Multiple Seeds

```bash
for seed in 42 123 2024 456 789; do
  python -m emm_dti.train --data_dir data/human --epochs 200 --seed $seed
done
```

### View Results

```bash
python aggregate_results.py --dataset enzyme
python aggregate_results.py --dataset human
```

## Reproducibility

To ensure reproducible results:

1. Use the exact same hyperparameters specified in `configs/`
2. Use the machine configuration saved in `machine_config.json`
3. Set random seeds (handled automatically in training scripts)
4. Use the same dataset versions

## Troubleshooting

### CUDA Out of Memory (OOM)

Reduce batch size:
```bash
python -m emm_dti.train --data_dir data/human --batch_size 8
```

### RDKit Import Error

Install via conda (more stable):
```bash
conda install -c conda-forge rdkit
```

### Dataset Not Found

Ensure data files are in the correct directory:
```bash
ls data/enzyme/  # Should show: drugs.csv, proteins.csv, interactions.csv
```

## Hardware Configuration

This code was tested on:
- **CPU**: AMD Ryzen 5 (6 cores / 12 logical cores @ 3.30 GHz)
- **GPU**: NVIDIA GeForce GTX 1650 (4 GB VRAM, compute capability 7.5)
- **RAM**: 15.35 GB
- **OS**: Windows 11
- **CUDA**: 12.8
- **cuDNN**: 8.9.0

See `machine_config.json` for your system's configuration.

## Citation

If you use this code, please cite:

```bibtex
@article{emm_dti,
  title={EMM-DTI: Enhanced Mamba with Multi-headed attention for Drug-Target Interaction prediction},
  author={...},
  year={2026}
}
```
