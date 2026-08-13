# FCS Mining in EMM-DTI

## Overview

**FCS (Frequent Continuous Subsequence) mining** is a core component of EMM-DTI that extracts meaningful fragments from drug (SMILES) and protein sequences.

Instead of treating sequences as raw characters, FCS discovers **frequent, recurring patterns** that are likely to be functionally important.

---

## How It Works

### 1️⃣ **Pattern Discovery (Apriori Algorithm)**

Starting with a database of SMILES and protein sequences:

```
Input Sequences:
  SMILES:  "CC(=O)Oc1ccccc1", "CN1C(=O)CC(c2ccccc2)C1=O", ...
  Proteins: "MSVPTSSMFFHQSN...", "MEYFTVGYPPN...", ...

↓

FCS Mining (min_support=0.3)
  Find all 1-mers (k=1) appearing in ≥30% of sequences
  Find all 2-mers (k=2) appearing in ≥30% of sequences  
  Find all 3-mers (k=3) appearing in ≥30% of sequences
  ...

Output Frequent Patterns:
  1-mers: {'C', 'O', '=', '(', ')', 'M', 'S', 'V', 'P', ...}
  2-mers: {'c1', 'cc', '(=', '=O', 'MS', 'SV', 'VP', ...}
  3-mers: {'c1c', 'ccc', '(=O', 'MSV', 'SVP', 'VPT', ...}
```

### 2️⃣ **Vocabulary Building**

Frequent patterns are mapped to indices for embedding lookup:

```python
Fragment → Index
  <PAD>     → 0      (special: padding)
  <UNK>     → 1      (special: unknown)
  'C'       → 2
  'O'       → 3
  '='       → 4
  '(='      → 5
  '=O'      → 6
  '(=O'     → 7
  'M'       → 8
  'S'       → 9
  'V'       → 10
  ...
```

### 3️⃣ **Sequence Tokenization (Greedy Matching)**

When processing a sequence, use **longest-match-first** greedy algorithm:

```
Sequence: "CC(=O)Oc1ccccc1"

Tokenization Process:
  Position 0-1: "CC" → Check if "CC" in patterns? No
  Position 0:   "C"  → Check if "C" in patterns? Yes! → Index 2
                       Move to position 1
  
  Position 1-2: "C(" → Check if "C(" in patterns? No
  Position 1:   "C"  → Check if "C" in patterns? Yes! → Index 2
                       Move to position 2
  
  Position 2-4: "(=O" → Check if "(=O" in patterns? Yes! → Index 7
                        Move to position 5
  
  Position 5-6: "Oc" → Check if "Oc" in patterns? No
  Position 5:   "O"  → Check if "O" in patterns? Yes! → Index 3
                       Move to position 6
  ...

Result: [2, 2, 7, 3, 10, 10, 10, 10, 10, 1]  (padded to max_length)
```

---

## Implementation Details

### Code Flow

```python
# 1. Load data
data_module = DTIDataModule(data_dir="data/human")
    ├─ Loads drugs.csv, proteins.csv, interactions.csv
    └─ Calls _create_fcs_vocabulary()
         ├─ FCSModule.mine(all_sequences, max_k=3)
         │   └─ Uses Apriori algorithm to find frequent k-mers
         └─ FragmentVocabulary.build_from_fcs(fcs)
            └─ Creates index mapping for all patterns

# 2. Create DataLoaders
train_loader, val_loader, test_loader = data_module.create_loaders()
    └─ DTIDataset receives:
         ├─ drug_sequences (SMILES strings)
         ├─ protein_sequences (amino acid strings)
         ├─ fcs_vocab (vocabulary mapping)
         └─ fcs_patterns (mined patterns by length)

# 3. Process sequences during training
for drug_indices, protein_indices, labels in train_loader:
    ├─ drug_indices = DTIDataset._sequence_to_indices(smiles, max_len=100)
    │   └─ Greedy matching against fcs_patterns
    └─ protein_indices = DTIDataset._sequence_to_indices(sequence, max_len=200)
        └─ Greedy matching against fcs_patterns

# 4. Model inference
model(drug_indices, protein_indices)
    ├─ Embedding layer: indices → 128-dim vectors
    ├─ Mamba layers: sequence → context-aware representations
    ├─ Interaction matrix: compute drug-protein affinity
    └─ CNN + MLP: predict DTI probability
```

---

## Configuration

### Adjust FCS Mining Parameters

In `emm_dti/data/loaders.py`, line ~180:

```python
# Current setting: min_support=0.3 (patterns in ≥30% of sequences)
self.fcs = FCSModule(min_support=0.3)
patterns = self.fcs.mine(all_sequences, max_k=3)

# To increase/decrease pattern diversity:
# Higher min_support (0.5) → Fewer, more common patterns → Smaller vocab
# Lower min_support (0.1)  → More patterns, better coverage → Larger vocab
```

### Adjust Greedy Matching Strategy

In `emm_dti/data/loaders.py`, line ~120-140:

Current: **Longest-match-first** (greedy for longest patterns)
```python
# Try longest patterns first
for pattern in sorted(self.patterns_by_len.keys(), reverse=True):
    if sequence[i:].startswith(pattern):
        indices.append(...)
        i += len(pattern)
        break
```

Alternative: **Highest-frequency-first** (not currently implemented)
```python
# Could rank by support count instead of length
```

---

## Performance Impact

### Vocabulary Size vs. Model Size

```
FCS min_support=0.3  →  ~500-1000 tokens  →  Embedding dim 128  →  ~64K params
FCS min_support=0.1  →  ~2000-5000 tokens →  Embedding dim 128  →  ~256K params
FCS min_support=0.5  →  ~100-200 tokens   →  Embedding dim 128  →  ~12K params
```

### Training Speed

- **Smaller vocab**: Faster embedding lookup, but less expressive
- **Larger vocab**: Slower embedding, but captures more patterns
- **Optimal**: ~500-1000 tokens for balance

---

## Example: Aspirin Tokenization

### Input
```
SMILES: "CC(c=o)c1ccccc1c(=o)O"
```

### FCS Patterns Found
```
1-mers: C, (, c, =, o, ), 1, c, c, c, c, c, c, 1, c, (, =, o, ), O
2-mers: CC, C(, (c, c=, =o, o), )c, c1, 1c, cc, c1, (=, =o, o)
3-mers: CC(, C(c, (c=, c=o, =o), o)c, )c1, c1c, 1cc, cc1, c1c, (=o, =o)
```

### Tokenization (Greedy)
```
Position  Substring  Match        Token   Index
0-2       "CC("      No → "C"     C       2
1-2       "C("       No → "C"     C       2
2-4       "(c="      No → "(c"    No → "(" ?
3-5       "c=o"      No → "c"     c       10
4-6       "=o)"      No → "="     =       4
5-7       "o)c"      No → "o"     o       3
6-8       ")c1"      No → ")"     )       ?
7-9       "c1c"      Yes! "(=o"   (=o     7
...
```

### Output Tokens
```
[2, 2, 5, 10, 4, 3, 6, 10, 10, 10, 10, 10, 10, 9, 10, ...]
 C  C  (=  c  =  o  )c  c  c  c  c  c  c  c  1   ...
```

---

## Validation

### Check FCS Mining Output

```python
# After data module initialization
from emm_dti.data import DTIDataModule

dm = DTIDataModule("data/human")

# View mined patterns
print(f"Total patterns: {len(dm.fcs)}")
print(f"1-mers: {len(dm.fcs.get_patterns(1))}")
print(f"2-mers: {len(dm.fcs.get_patterns(2))}")
print(f"3-mers: {len(dm.fcs.get_patterns(3))}")

# View vocabulary
print(f"Vocabulary size: {len(dm.fcs_vocab)}")
print(f"First 20 tokens: {[dm.fcs_vocab.get_fragment(i) for i in range(20)]}")

# Check specific pattern support
pattern_support = dm.fcs.get_support("c1c")
print(f"Support for 'c1c': {pattern_support} sequences")
```

### Debug Tokenization

```python
# Test tokenization on a sample
dataset = dm.create_loaders()[0].dataset
smiles = "CC(=O)Oc1ccccc1C(=O)O"
indices = dataset._sequence_to_indices(smiles, max_len=100)
print(f"SMILES: {smiles}")
print(f"Indices: {indices[:20]}")  # First 20 tokens
```

---

## Summary

| Stage | Component | Input | Output |
|-------|-----------|-------|--------|
| **Mining** | FCSModule + Apriori | Sequences | Frequent k-mers |
| **Vocab** | FragmentVocabulary | Patterns | Index mapping |
| **Tokenization** | Greedy matching | SMILES/Sequence | Token indices |
| **Embedding** | nn.Embedding | Indices | Dense vectors |
| **Model** | EMMDTI | Vectors | DTI prediction |

FCS mining enables the model to learn from **meaningful chemical and biological units** rather than raw characters, leading to better generalization and interpretability.
