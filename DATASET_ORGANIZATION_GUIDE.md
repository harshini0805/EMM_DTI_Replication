# Dataset Organization Guide

This guide will organize all your datasets into the standard EMM-DTI format.

---

## 📊 Datasets You Have

```
Root folder:
├── human/                  ✅ Already organized in data/human/
├── humans_random/          ⏳ Will organize
├── celegans_random/        ⏳ Will organize  
├── bindingdb_random/       ⏳ Will organize
├── drugbank/               ⏳ Will organize
└── enzyme/                 ⏳ Will organize
```

---

## 🚀 Step 1: Run Organization Script

```bash
cd D:\Projects\EMM_DTI_Replication

# Activate venv
source venv/bin/activate

# Run organization
python organize_all_datasets.py
```

**What it does:**
1. Combines train/test/valid splits into single datasets
2. Creates standard format: `data/<dataset>/drugs.csv`, `proteins.csv`, `interactions.csv`
3. Auto-detects SMILES, sequence, and interaction columns
4. Verifies each dataset after organization

**Expected output:**
```
================================================================================
EMM-DTI Multi-Dataset Organization Tool
================================================================================

Organizing humans_random dataset...
  Loaded train: XXX samples
  Loaded test: XXX samples
  Loaded valid: XXX samples
  ✅ Created drugs.csv (XXX unique drugs)
  ✅ Created proteins.csv (XXX unique proteins)
  ✅ Created interactions.csv (XXX total)
    - Positive: XXX
    - Negative: XXX
```

---

## 🔍 Step 2: Verify Organization

After running the script, check:

```bash
ls -la data/

# Should show:
# data/human/              ✅
# data/humans_random/      ✅
# data/celegans_random/    ✅
# data/bindingdb_random/   ✅
# data/drugbank/           ✅
# data/enzyme/             ✅
```

Each should have:
```bash
ls data/humans_random/
# drugs.csv
# proteins.csv
# interactions.csv
```

---

## ⚙️ Step 3: Create Config Files

Once organized, create config for each dataset. **Example for humans_random:**

**File:** `configs/train_humans_random_ssm.yaml`

```yaml
dataset:
  name: humans_random
  data_dir: data/humans_random
  train_split: 0.7
  val_split: 0.2
  test_split: 0.1
  random_seed: 42

model:
  type: emm_dti_ssm
  fcs_embedding_dim: 128
  mamba_hidden_dim: 256
  mamba_n_layers: 2
  mamba_state_size: 16
  cnn_out_channels: 3
  cnn_kernel_size: 3
  dropout: 0.1

training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 200
  early_stopping_patience: 30
  device: cuda

optimization:
  optimizer: adam
  weight_decay: 1.0e-05
  gradient_clip: 1.0
  lr_scheduler: none

logging:
  log_dir: results/humans_random_ssm
  log_level: INFO
```

**Create configs for all:**
```bash
# Copy template and edit each one
cp configs/train_human_ssm.yaml configs/train_humans_random_ssm.yaml
cp configs/train_human_ssm.yaml configs/train_celegans_random_ssm.yaml
cp configs/train_human_ssm.yaml configs/train_bindingdb_random_ssm.yaml
cp configs/train_human_ssm.yaml configs/train_drugbank_ssm.yaml
cp configs/train_human_ssm.yaml configs/train_enzyme_ssm.yaml

# Edit each one to change:
# - data_dir: data/<dataset_name>
# - log_dir: results/<dataset_name>_ssm
```

---

## 🧪 Step 4: Test Each Dataset

Once configs are created, test training on each:

```bash
# Test humans_random
python -m emm_dti.train --config configs/train_humans_random_ssm.yaml --epochs 10

# Test celegans_random
python -m emm_dti.train --config configs/train_celegans_random_ssm.yaml --epochs 10

# Test bindingdb_random
python -m emm_dti.train --config configs/train_bindingdb_random_ssm.yaml --epochs 10

# Test drugbank
python -m emm_dti.train --config configs/train_drugbank_ssm.yaml --epochs 10

# Test enzyme
python -m emm_dti.train --config configs/train_enzyme_ssm.yaml --epochs 10
```

---

## 📊 Expected Dataset Sizes

Based on original papers:

| Dataset | Drugs | Proteins | Interactions | Positive | Negative |
|---------|-------|----------|--------------|----------|----------|
| **human** | 2,726 | 2,001 | 6,728 | 3,364 | 3,364 |
| **humans_random** | ? | ? | ? | ? | ? |
| **celegans_random** | ? | ? | ? | ? | ? |
| **bindingdb_random** | ? | ? | ? | ? | ? |
| **drugbank** | ? | ? | ? | ? | ? |
| **enzyme** | ? | ? | ? | ? | ? |

After running `organize_all_datasets.py`, you'll see actual numbers.

---

## 🎯 Full Workflow

```bash
# 1. Organize all datasets
python organize_all_datasets.py

# 2. Create configs for each
cp configs/train_human_ssm.yaml configs/train_humans_random_ssm.yaml
# ... edit as needed

# 3. Run full ensemble on each (optional)
# Edit ensemble_train_ssm.py to use different datasets
python ensemble_train_ssm.py

# 4. Compare results across datasets
ls results/*/ensemble_results.json
```

---

## ⚠️ Troubleshooting

### Issue: "No CSV files found"
- Check that folders have the expected structure
- Run `ls -la` to see what's actually in the folder

### Issue: "Could not auto-detect columns"
- Edit `organize_all_datasets.py` lines 96-108
- Specify column names manually:
  ```python
  smiles_col = 'drug_smiles'  # Change to actual column name
  seq_col = 'target_seq'
  inter_col = 'binary_label'
  ```

### Issue: "Datasets look imbalanced"
- This is OK! Different datasets have different characteristics
- Just note the balance ratio in your report

---

## ✅ Checklist

- [ ] Run `python organize_all_datasets.py`
- [ ] Verify `ls data/` shows all 6 datasets
- [ ] Create config files for each dataset
- [ ] Test training: `python -m emm_dti.train --config configs/train_humans_random_ssm.yaml --epochs 10`
- [ ] All configs work without errors
- [ ] Ready to run full ensemble

---

## 🚀 Next Steps

Once all datasets are organized and configs created:

1. **Run quick tests** on each dataset (10-20 epochs)
2. **Compare dataset sizes** and characteristics
3. **Run full ensemble** on datasets of interest
4. **Report results** with dataset comparison table

Good luck! 🎯
