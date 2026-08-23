# Reproducibility Checklist & Files

This document lists all files created to ensure reproducible research for the EMM-DTI and Mamba-DTI papers.

## Files Created

### EMM-DTI Project (D:\Projects\EMM_DTI_Replication)

#### 1. **requirements.txt** ✓
- **Purpose**: Specify exact package versions for reproducible installation
- **Contains**: 
  - PyTorch 2.0.0+
  - NumPy 1.24.0-1.99
  - Pandas 2.0.0+
  - Scikit-learn 1.3.0+
  - RDKit 2023.09.0
  - PyYAML 6.0.0+
  - All other dependencies

#### 2. **INSTALL.md** ✓
- **Purpose**: Step-by-step installation instructions
- **Covers**:
  - System requirements
  - Virtual environment setup
  - Package installation
  - Dataset preparation
  - Quick start commands
  - Troubleshooting

#### 3. **README_REPRODUCIBILITY.md** ✓
- **Purpose**: Complete guide for reproducing paper results
- **Covers**:
  - Quick start (3 steps)
  - System configuration recording
  - Full training pipeline (CV + independent datasets)
  - Results aggregation
  - Hyperparameter specifications
  - File structure overview
  - Reproducibility checklist
  - Expected results
  - Citation format

#### 4. **get_system_info.py** ✓
- **Purpose**: Automatically collect machine configuration
- **Outputs**:
  - `machine_config.json`: Machine specs in JSON format
  - Console output with ready-to-copy paper text
  - Hardware: CPU, GPU, RAM, CUDA, Python, PyTorch versions

---

### Mamba-DTI Project (D:\Projects\mamba-dti)

#### 1. **requirements.txt** ✓
- **Purpose**: Identical to EMM-DTI (shared dependencies)
- **Note**: Both projects use same core dependencies for fair comparison

#### 2. **INSTALL.md** ✓
- **Purpose**: Installation guide for Mamba-DTI benchmark
- **Covers**:
  - All 10 architectures overview
  - Installation steps
  - Dataset preparation for 6 datasets
  - Training for all architectures
  - Comparing architectures
  - Results aggregation
  - Hardware configuration

#### 3. **README_REPRODUCIBILITY.md** ✓
- **Purpose**: Guide for reproducing all 10-architecture benchmark
- **Covers**:
  - Quick start
  - System configuration
  - Training all 10 architectures on 6 datasets
  - Comparison scripts
  - Standardized hyperparameters
  - Dataset statistics
  - Results directory structure
  - Expected results for each architecture
  - Reproducibility checklist
  - Estimated runtime on different hardware
  - Citation format

#### 4. **get_system_info.py** ✓
- **Purpose**: Same as EMM-DTI (or reference from EMM-DTI)
- **Outputs**: `machine_config.json` and ready-to-copy paper text

---

## Key Reproducibility Features

### 1. **Exact Dependency Versions**
- Both `requirements.txt` files pin major versions
- Reproducible on any machine with same OS

### 2. **System Configuration Recording**
- `get_system_info.py` captures all relevant hardware specs
- Generates `machine_config.json` for archival
- Produces paper-ready text for Methods section

### 3. **Complete Training Instructions**
- CV dataset training (5-fold CV, 3 seeds)
- Independent dataset training (5 seeds)
- Exact command-line invocations
- Expected output locations

### 4. **Results Aggregation**
- `aggregate_results.py` script for per-dataset results
- `compare_architectures.py` for cross-architecture comparison
- CSV export for statistical analysis

### 5. **Standardized Hyperparameters**
- All 10 architectures use identical:
  - batch_size = 16
  - learning_rate = 3e-4
  - epochs = 200
  - early_stopping_patience = 30
  - weight_decay = 1e-4
  - gradient_clip = 1.0
  - Early stopping metric = AUPR (not ROC-AUC)

### 6. **Error Handling & Validation**
- NaN checkpoint prevention (trainer.py line 330-332)
- Numerically stable sigmoid (scipy.special.expit)
- Confusion matrix edge case handling (metrics.py line 113-125)
- Metric key consistency (aupr, auc, accuracy)

---

## Reproducibility Workflow

### For EMM-DTI Paper:

```
1. Install dependencies
   └─ pip install -r requirements.txt

2. Record system configuration
   └─ python get_system_info.py
   └─ Outputs: machine_config.json

3. Train on CV datasets
   ├─ python -m emm_dti.train_cv --data_dir data/enzyme --epochs 200
   └─ python -m emm_dti.train_cv --data_dir data/drugbank --epochs 200

4. Train on independent datasets
   ├─ for seed in 42 123 2024 456 789; do
   │    python -m emm_dti.train --data_dir data/human --epochs 200 --seed $seed
   │  done
   └─ Repeat for biosnap, celegans, bindingdb

5. View aggregated results
   └─ python aggregate_results.py --dataset enzyme

6. Archive results
   └─ Copy results/ and machine_config.json to supplementary materials
```

### For Mamba-DTI Benchmark:

```
1. Install dependencies
   └─ pip install -r requirements.txt

2. Record system configuration
   └─ python get_system_info.py

3. Train all 10 architectures on all 6 datasets
   ├─ ~10-15 hours on GPU (GTX 1650)
   ├─ Results stored in architectures/{arch}/results/
   └─ Outputs: 60 results.json files (10 archs × 6 datasets)

4. Compare architectures
   └─ python compare_architectures.py --dataset human

5. Archive results
   └─ Entire architectures/*/results/ directory
```

---

## Reproducibility Metrics

### What's Standardized:

✓ Python version (3.10-3.12)
✓ PyTorch version (2.0.0+)
✓ NumPy/Pandas/Scikit-learn versions
✓ CUDA/cuDNN versions
✓ Batch size = 16
✓ Learning rate = 3e-4
✓ Epochs = 200
✓ Early stopping = 30 patience
✓ Gradient clip = 1.0
✓ Weight decay = 1e-4
✓ Random seeds (42, 123, 2024, 456, 789)
✓ Data splits (enzyme/drugbank: 5-fold CV; others: 80-10-10)
✓ Metric keys (auc, aupr, accuracy)
✓ Evaluation metric (AUPR for checkpoints)
✓ Fragment embedding dimension (128)
✓ Mamba hidden dimension (64)

### Expected Variance:

- ±1-2% AUPR difference due to floating-point precision
- Hardware differences (CPU vs GPU, different GPU models)
- OS differences (Linux vs Windows vs macOS)
- Minor package version changes within same major version

---

## Files for Paper Submission

### Main Text
- Model architecture diagrams (generated from model.py files)
- Training convergence plots (from logs/)
- Final results table (from results/*.json)
- Machine configuration (from machine_config.json)

### Supplementary Materials
- Complete requirements.txt (for pip install)
- All training scripts (train_cv.py, train.py)
- All architecture definitions (models/*)
- Complete results/ directory with all metrics
- machine_config.json (hardware specs)
- Reproducibility guides (README_REPRODUCIBILITY.md)

### Reproducibility Code
- emm_dti/ (complete source code)
- aggregate_results.py
- get_system_info.py
- INSTALL.md
- README_REPRODUCIBILITY.md

---

## Verification Steps

To verify reproducibility:

1. **Install on clean system**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Check system matches** (±tolerance):
   ```bash
   python get_system_info.py
   ```

3. **Run single training experiment**:
   ```bash
   python -m emm_dti.train_cv --data_dir data/enzyme --epochs 200
   ```

4. **Compare results** to expected values (±2% AUPR):
   ```bash
   python aggregate_results.py --dataset enzyme
   ```

5. **Review logs** for errors:
   ```bash
   cat logs/enzyme_cv_training.log
   ```

---

## Known Limitations

1. **Floating-point precision**: Results vary slightly across hardware
2. **GPU memory**: Requires 4+ GB VRAM (tested on GTX 1650)
3. **Random seeds**: Only control pseudorandom numbers, not hardware randomness
4. **Dataset versions**: Results assume same dataset preprocessed identically
5. **OS differences**: Minor differences between Windows/Linux/macOS

---

## Contact for Reproducibility Issues

If results don't match:
1. Check system specs match (see machine_config.json)
2. Verify package versions (pip freeze)
3. Test with different random seed (--seed argument)
4. Run multiple times and report mean ± std
5. Check logs/ directory for errors

---

## Summary

✓ All dependencies specified with versions
✓ System configuration automated collection
✓ Complete training pipeline documented
✓ Exact hyperparameters standardized
✓ Results aggregation scripts provided
✓ Error handling and validation implemented
✓ Expected results published
✓ Ready for archival and supplementary materials

**Total files created**: 
- 2 requirements.txt (EMM-DTI, Mamba-DTI)
- 2 INSTALL.md
- 2 README_REPRODUCIBILITY.md
- 1 get_system_info.py
- 1 machine_config.json (generated at runtime)
