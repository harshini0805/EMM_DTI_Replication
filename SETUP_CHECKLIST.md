# Setup Checklist

Complete these steps to get EMM-DTI running:

## ✅ Prerequisites
- [ ] Python 3.9+ installed
- [ ] pip package manager working
- [ ] 8GB+ RAM available
- [ ] GPU with 8GB+ VRAM (optional, CPU works but slower)

## ✅ Installation

- [ ] Navigate to project directory:
  ```bash
  cd C:\Users\Harshini J\Engineering\Projects\EMM_DTI_Replication
  ```

- [ ] Create virtual environment:
  ```bash
  python -m venv venv
  source venv/Scripts/activate  # Windows: venv\Scripts\activate
  ```

- [ ] Install package:
  ```bash
  pip install -e .
  ```

- [ ] Verify installation:
  ```bash
  python -c "from emm_dti import EMMDTI; print('✓ EMM-DTI installed')"
  ```

## ✅ Data Preparation

- [ ] Verify data structure:
  ```bash
  ls data/human/
  ```
  Should show:
  - `drugs.csv`
  - `proteins.csv`
  - `interactions.csv`

- [ ] Check data format (first few lines):
  ```bash
  head -3 data/human/drugs.csv
  head -3 data/human/proteins.csv
  head -3 data/human/interactions.csv
  ```

- [ ] Verify data counts:
  ```bash
  python -c "import pandas as pd; print('Drugs:', len(pd.read_csv('data/human/drugs.csv'))); print('Proteins:', len(pd.read_csv('data/human/proteins.csv'))); print('Interactions:', len(pd.read_csv('data/human/interactions.csv')))"
  ```

## ✅ Configuration

- [ ] Check config file exists:
  ```bash
  ls configs/train_human.yaml
  ```

- [ ] Review configuration:
  ```bash
  cat configs/train_human.yaml
  ```

- [ ] Create results directory:
  ```bash
  mkdir -p results/human
  ```

## ✅ Device Setup

- [ ] Check GPU availability (if applicable):
  ```bash
  python -c "import torch; print('GPU Available:', torch.cuda.is_available())"
  ```

- [ ] Test device usage:
  ```bash
  python -c "from emm_dti.utils import print_device_info; print_device_info()"
  ```

## ✅ Quick Test (Optional but Recommended)

- [ ] Run test on small batch to verify everything works:
  ```bash
  python -m emm_dti.train --config configs/train_human.yaml --epochs 2 --batch_size 16
  ```

  This should:
  - Load data in ~10-30 seconds
  - Run 2 epochs
  - Save a checkpoint to `results/human/best_model.pt`

## ✅ Ready to Train!

If all checkboxes are complete, you're ready:

```bash
# Full training
python -m emm_dti.train --config configs/train_human.yaml
```

## 🔧 Troubleshooting

### Import Error: "ModuleNotFoundError"
```bash
# Reinstall with dependencies
pip install -e ".[dev]"
```

### CUDA Error: "out of memory"
```bash
# Use smaller batch size
python -m emm_dti.train --config configs/train_human.yaml --batch_size 16
```

### File Not Found: "data/human/drugs.csv"
```bash
# Verify data location
ls -la data/human/
# Files should exist in this exact location
```

### "No module named 'rdkit'"
```bash
pip install rdkit
```

### Training very slow on CPU
- This is normal (10-100x slower than GPU)
- Consider using cloud GPU services:
  - Google Colab (free)
  - AWS EC2 (GPU instances)
  - Azure ML (compute instances)

## 📊 Expected Training Time

| Device | Speed | Time (100 epochs) |
|--------|-------|------------------|
| NVIDIA RTX 3090 | Baseline | ~15-20 min |
| NVIDIA RTX 4090 | 1.3x faster | ~12-15 min |
| CPU (16 cores) | 10-50x slower | 2.5-8 hours |
| Google Colab | Similar to RTX 3090 | ~15-20 min |

## 📝 Next Actions

After training completes:

1. **Evaluate model:**
   ```bash
   python -m emm_dti.evaluate \
       --checkpoint results/human/best_model.pt \
       --data_dir data/human
   ```

2. **View results:**
   ```bash
   cat results/human/evaluation_results.json
   ```

3. **Analyze metrics:**
   - Open `results/human/evaluation_results.json`
   - Compare with Table 3 in README.md
   - Note: Slight variations expected (±0.005 AUC)

4. **Fine-tune (optional):**
   - Adjust `configs/train_human.yaml`
   - Try different hyperparameters
   - Retrain with better settings

---

**Status:** ✅ Ready to run  
**Last Updated:** 2026-07-24  
**Support:** See README.md for detailed documentation
