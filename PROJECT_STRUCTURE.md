# Project Structure: EMM-DTI Implementation

```
D:\EMM_DTI_Replication\EMM_DTI_Replication\
│
├── 📋 Configuration Files
│   ├── configs/
│   │   ├── train_human.yaml          ← All hyperparameters (paper-matching)
│   │   ├── default.yaml              ← Base config
│   │   └── train_human.yaml
│   │
│   └── setup.py                      ← Package installation
│
├── 🧠 Core Model Architecture
│   └── emm_dti/
│       ├── models/
│       │   ├── emm_dti.py            ← MAIN MODEL (220 lines)
│       │   │   ├── EMMDTI class (lines 18-224)
│       │   │   │   ├── Embedding + LayerNorm (lines 63-65)
│       │   │   │   ├── Drug Mamba (lines 68-74)
│       │   │   │   ├── Protein Mamba (lines 77-83)
│       │   │   │   ├── Interaction CNN (lines 88-92)
│       │   │   │   ├── Predictor MLP (lines 96-105)
│       │   │   │   └── Forward pass (lines 125-170)
│       │   │
│       │   ├── mamba.py              ← MAMBA STATE SPACE MODEL (206 lines)
│       │   │   ├── MambaLayer class (lines 15-111)
│       │   │   │   ├── A, B, C matrices (lines 49-56)
│       │   │   │   ├── State transition h_t = A·h_{t-1} + B·x_t (line 93)
│       │   │   │   └── Output y_t = C·h_t (line 96)
│       │   │   │
│       │   │   └── BidirectionalMamba class (lines 113-206)
│       │   │       ├── Forward direction (lines 184-186)
│       │   │       ├── Backward direction (lines 189-192)
│       │   │       └── Output projection (line 195)
│       │   │
│       │   └── fcs.py                ← FCS MINING MODULE (224 lines)
│       │       ├── FCSModule class (lines 14-153)
│       │       │   ├── Apriori algorithm mine() (lines 56-116)
│       │       │   ├── K-mer extraction (lines 40-54)
│       │       │   └── Support threshold filtering (lines 75-100)
│       │       │
│       │       └── FragmentVocabulary class (lines 155-224)
│       │           └── Pattern→Index mapping for embeddings
│       │
│       ├── 📊 Data Pipeline
│       └── data/
│           ├── loaders.py            ← DATA LOADING (400 lines)
│           │   ├── DTIDataset class (lines 20-140)
│           │   │   ├── Sequence to indices (lines 99-139)
│           │   │   └── Greedy pattern matching
│           │   │
│           │   └── DTIDataModule class (lines 142-400)
│           │       ├── Data loading (lines 210-239)
│           │       ├── 7:2:1 split (lines 193-206)
│           │       ├── FCS mining from training data (lines 268-276)
│           │       ├── Data leakage prevention ✓
│           │       └── DataLoader creation (lines 310-365)
│           │
│           ├── preprocessing.py      ← PREPROCESSING (199 lines)
│           │   ├── SMILES canonicalization
│           │   ├── Protein sequence validation
│           │   └── DataPreprocessor class
│           │
│           └── __init__.py
│
├── 🎓 Training & Evaluation
│   └── training/
│       ├── trainer.py                ← TRAINING LOOP (350+ lines)
│       │   ├── Trainer class (lines 24+)
│       │   │   ├── Optimizer setup (lines 60-100)
│       │   │   ├── Scheduler setup (lines 102-134)
│       │   │   ├── Train epoch (lines 136-201)
│       │   │   │   ├── Forward pass
│       │   │   │   ├── Backward pass
│       │   │   │   ├── Gradient clipping (lines 178-179)
│       │   │   │   └── Sigmoid conversion (line 196)
│       │   │   │
│       │   │   ├── Validation (lines 203-250)
│       │   │   │   └── Sigmoid conversion (line 247)
│       │   │   │
│       │   │   ├── Early stopping (lines 268-285)
│       │   │   ├── Checkpointing (lines 301-320)
│       │   │   └── Fit method (lines 258-350+)
│       │   │
│       │   ├── predict() method
│       │   └── Loss function: BCEWithLogitsLoss (line 292)
│       │
│       ├── metrics.py                ← EVALUATION METRICS (248 lines)
│       │   ├── Metrics class (lines 24-181)
│       │   │   ├── AUC (line 69)
│       │   │   ├── AUPR (lines 76-77)
│       │   │   ├── Accuracy (line 83)
│       │   │   ├── Precision (line 87)
│       │   │   ├── Recall (line 94)
│       │   │   ├── F1-Score (lines 99-110)
│       │   │   ├── Specificity (lines 119-123)
│       │   │   ├── MCC (lines 128-133)
│       │   │   └── TP, TN, FP, FN counts (lines 112-117)
│       │   │
│       │   └── MetricsTracker class (lines 183-247)
│       │       └── Tracks metrics over epochs
│       │
│       └── __init__.py
│
├── 🛠️ Utilities
│   └── utils/
│       ├── config.py                 ← Configuration loading
│       ├── logging_utils.py          ← Logging setup
│       ├── device.py                 ← GPU/CPU device handling
│       └── __init__.py
│
├── 📚 Ensemble Training
│   ├── ensemble_train.py             ← ENSEMBLE (306 lines)
│   │   ├── Multiple seeds [42, 123, 2024, 456, 789]
│   │   ├── train_seed() function (lines 103-186)
│   │   ├── Prediction averaging (line 232)
│   │   └── Error bars (mean ± std) (lines 245-260)
│   │
│   ├── run_ensemble.ps1              ← PowerShell launcher
│   ├── run_ensemble.bat              ← Batch launcher
│   └── run_ensemble.sh               ← Linux/macOS launcher (tmux)
│
├── 📖 Documentation
│   ├── IMPLEMENTATION_MAP.md         ← Paper → Code mapping ✓ YOU ARE HERE
│   ├── ENSEMBLE_TRAINING_README.md   ← How to run ensemble
│   ├── VERIFICATION_REPORT.md        ← Correctness verification
│   └── PROJECT_STRUCTURE.md          ← This file
│
└── 📊 Data & Results
    ├── data/
    │   └── human/
    │       ├── drugs.csv              ← 2,726 drugs with SMILES
    │       ├── proteins.csv           ← 2,001 proteins with sequences
    │       ├── interactions.csv       ← 6,728 interactions (3,364+3,364)
    │       └── .fcs_cache.pkl         ← Cached FCS patterns (reproducibility)
    │
    └── results/
        ├── human/                     ← Single run results
        │   ├── config.yaml
        │   └── ...
        │
        └── ensemble/                  ← Multi-seed ensemble results
            ├── ensemble_results.json   ← Final AUC ± std
            ├── training.log            ← Full training output
            ├── seed_42/
            │   ├── best_model.pt
            │   └── results.json
            ├── seed_123/
            │   ├── best_model.pt
            │   └── results.json
            ├── seed_2024/
            │   ├── best_model.pt
            │   └── results.json
            ├── seed_456/
            ├── seed_789/
            └── ...
```

---

## File Purpose Summary

### 🔴 CRITICAL FILES (Paper Implementation)

| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `emm_dti/models/emm_dti.py` | Main model architecture | 220 | ✅ Complete |
| `emm_dti/models/mamba.py` | Mamba state-space layers | 206 | ✅ Correct |
| `emm_dti/models/fcs.py` | FCS mining algorithm | 224 | ✅ Verified |
| `emm_dti/data/loaders.py` | Data loading & splitting | 400 | ✅ Verified |
| `emm_dti/training/trainer.py` | Training loop | 350+ | ✅ Verified |
| `emm_dti/training/metrics.py` | Evaluation metrics | 248 | ✅ Verified |
| `configs/train_human.yaml` | Hyperparameters | 46 | ✅ Verified |

### 🟡 ENSEMBLE ENHANCEMENT

| File | Purpose | Lines |
|------|---------|-------|
| `ensemble_train.py` | Multi-seed averaging | 306 |
| `run_ensemble.ps1` | Windows launcher | 124 |
| `run_ensemble.bat` | Batch launcher | 51 |
| `run_ensemble.sh` | Linux/macOS launcher | 128 |

### 🔵 DOCUMENTATION

| File | Purpose |
|------|---------|
| `IMPLEMENTATION_MAP.md` | Paper → Code mapping |
| `PROJECT_STRUCTURE.md` | File organization (this file) |
| `ENSEMBLE_TRAINING_README.md` | Usage instructions |
| `VERIFICATION_REPORT.md` | Correctness checks |

---

## How to Show This to Faculty

### Option 1: Live Walkthrough
```bash
# Show the structure
ls -la D:\EMM_DTI_Replication\EMM_DTI_Replication\emm_dti\models\

# Show FCS mining
code emm_dti/models/fcs.py

# Show Mamba implementation
code emm_dti/models/mamba.py

# Show main model
code emm_dti/models/emm_dti.py

# Show training loop
code emm_dti/training/trainer.py

# Show metrics
code emm_dti/training/metrics.py

# Show config
code configs/train_human.yaml
```

### Option 2: Print This Document
```
IMPLEMENTATION_MAP.md          ← Show faculty this
PROJECT_STRUCTURE.md           ← Show this too
```

### Option 3: Key Takeaways to Highlight

✅ **Paper Implementation:**
- FCS mining with Apriori algorithm (fcs.py)
- Bidirectional Mamba SSM layers (mamba.py)
- Interaction matrix via dot product (emm_dti.py)
- CNN feature extraction 3×3 kernel (emm_dti.py)
- MLP predictor (emm_dti.py)
- Binary cross-entropy loss (trainer.py)
- 5 evaluation metrics (metrics.py)

✅ **Data Integrity:**
- 7:2:1 train/val/test split (loaders.py)
- FCS mining from training data only (prevents leakage)
- Verified 6,728 samples (3,364 pos + 3,364 neg)

✅ **Enhancement:**
- Multi-seed ensemble (5 runs)
- Error bars (mean ± std)
- Better than paper's single-run approach

---

## Code Statistics
```
Total Lines:       ~3,000
Core Model:        ~500
Data Pipeline:     ~400
Training Loop:     ~350
Metrics:           ~250
FCS Mining:        ~230
Mamba Layers:      ~210
Config:            ~50
Ensemble:          ~300
Launchers:         ~300
```
