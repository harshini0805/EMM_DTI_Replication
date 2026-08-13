# FCS Workflow: Do You Need Pre-computation?

## Short Answer: **NO** ❌

FCS patterns are mined **automatically** during data loading. You don't need to pre-generate them.

Just run:
```bash
python -m emm_dti.train --config configs/train_human.yaml
```

That's it. FCS mining happens internally (takes ~2-3 seconds).

---

## If You Want Reproducibility: Pre-compute ✅

For reproducible experiments (always same patterns), optionally pre-compute:

```bash
# One-time: Mine and cache patterns
python -m emm_dti.utils.fcs_precompute --data_dir data/human

# Creates: data/human/.fcs_cache.pkl

# Now training will reuse cached patterns:
python -m emm_dti.train --config configs/train_human.yaml
```

---

## Comparison: Automatic vs. Pre-computed

| Aspect | Automatic (Default) | Pre-computed (Optional) |
|--------|-------------------|------------------------|
| **Setup required?** | No | `python -m emm_dti.utils.fcs_precompute` |
| **Reproducibility** | Variable (depends on order) | Guaranteed (fixed cache) |
| **Speed** | 2-3 sec per run | Instant (loaded from cache) |
| **Memory** | Minimal | Minimal (cache is small) |
| **Use case** | Exploration/testing | Production/papers |

---

## Workflow Options

### Option 1: Quick & Easy (DEFAULT) ⚡
```bash
python -m emm_dti.train --config configs/train_human.yaml
# Automatic mining: ~2-3 sec overhead
# Training: ~15-20 min on GPU
# Total: ~15-20 min
```

### Option 2: Reproducible Experiments 🔒
```bash
# Step 1: Pre-compute (one-time, ~5 sec)
python -m emm_dti.utils.fcs_precompute --data_dir data/human

# Step 2: Train (uses cache)
python -m emm_dti.train --config configs/train_human.yaml
python -m emm_dti.train --config configs/train_human.yaml  # Identical patterns
python -m emm_dti.train --config configs/train_human.yaml  # Identical patterns
```

### Option 3: Custom FCS Parameters 🔧
```bash
# Mine with different thresholds
python -m emm_dti.utils.fcs_precompute \
    --data_dir data/human \
    --min_support 0.2 \
    --max_k 4
    
# Then train (will use this cache)
python -m emm_dti.train --config configs/train_human.yaml
```

---

## How Caching Works

### First Run (No Cache)
```
data/human/
├── drugs.csv
├── proteins.csv
├── interactions.csv
└── (no cache yet)

↓ python -m emm_dti.train

Loads data → Mines FCS patterns → Trains model
            ↓ (saves to cache)
            
data/human/
├── drugs.csv
├── proteins.csv
├── interactions.csv
└── .fcs_cache.pkl  ← Created!
```

### Second Run (Cache Exists)
```
data/human/
├── drugs.csv
├── proteins.csv
├── interactions.csv
└── .fcs_cache.pkl  ← Loaded instantly!

↓ python -m emm_dti.train

Loads data → Loads cache (instant) → Trains model
```

---

## Comparison with ESM/ChemBERTa

### ESM/ChemBERTa Workflow
```
# Pre-trained model (separate download/setup)
embeddings.pt  ← Pre-computed frozen embeddings

# Training
Load embeddings → Train classification head
(embeddings are fixed, not re-computed)
```

### EMM-DTI FCS Workflow
```
# No pre-training needed

# Training (Automatic)
Load data → Mine patterns (data-specific) → Train full model
           (patterns change based on data)

# Training (With Pre-computation)
Load data → Load cached patterns → Train full model
           (patterns reused, reproducible)
```

**Key difference**: ESM/ChemBERTa are **frozen pre-trained models**. FCS is **data-adaptive pattern mining**.

---

## Recommendations

| Scenario | Approach |
|----------|----------|
| **Exploring hyperparameters** | Automatic (fastest) |
| **Running ablations** | Pre-computed (reproducible) |
| **Writing paper** | Pre-computed (guaranteed same patterns) |
| **Production deployment** | Pre-computed (deterministic) |
| **One-off experiments** | Automatic (simplest) |

---

## Example: With/Without Cache

### Without Cache (Automatic - Default)
```bash
$ time python -m emm_dti.train --config configs/train_human.yaml

Loading data...
Mining FCS patterns from sequences...  # ← Takes 2-3 sec
FCS mining complete:
  1-mers: 24 patterns
  2-mers: 156 patterns
  3-mers: 289 patterns
Built vocabulary with 472 tokens from FCS patterns
Epoch 1/100
Train: auc: 0.6234 | aupr: 0.5891
...

real    15m23s  # Total time
```

### With Cache (Pre-computed)
```bash
# First time: Pre-compute
$ python -m emm_dti.utils.fcs_precompute --data_dir data/human
Mining FCS patterns from sequences...
FCS mining complete:
  1-mers: 24 patterns
  2-mers: 156 patterns
  3-mers: 289 patterns
Cached FCS patterns to data/human/.fcs_cache.pkl

# Second time: Use cache
$ time python -m emm_dti.train --config configs/train_human.yaml

Loading data...
Loading cached FCS patterns...  # ← Instant!
Loaded 469 patterns from cache
Epoch 1/100
Train: auc: 0.6234 | aupr: 0.5891
...

real    15m21s  # Same time, but reproducible patterns
```

---

## Summary

| Question | Answer |
|----------|--------|
| Do I need to pre-generate FCS? | **No** (automatic) |
| Will it slow down training? | **No** (2-3 sec overhead) |
| Can I cache for reproducibility? | **Yes** (optional) |
| How do I pre-compute? | `python -m emm_dti.utils.fcs_precompute --data_dir data/human` |
| Is it like ESM/ChemBERTa? | **No** (data-adaptive, not frozen) |

**Just run training. Everything happens automatically.** 🚀
