# EMM-DTI Complete Architecture Pipeline

**From Drug SMILES + Protein Sequence → DTI Prediction**

---

## 📊 COMPLETE DATA FLOW

```
Input: Drug SMILES + Protein Sequence
       ↓
[1] FCS MINING MODULE
       ↓
[2] FRAGMENT ENCODING
       ↓
[3] EMBEDDING LAYER
       ↓
[4] BIDIRECTIONAL MAMBA-SSM
       ↓
[5] SEQUENCE POOLING
       ↓
[6] INTERACTION MATRIX (Dot Product)
       ↓
[7] CNN FEATURE EXTRACTION
       ↓
[8] MLP PREDICTOR
       ↓
Output: DTI Prediction (Logit/Probability)
```

---

## 🔗 FILE REFERENCES FOR EACH STAGE

### **Stage 1: Data Loading & FCS Mining**

**File:** `emm_dti/data/loaders.py`

**Key Functions:**
- `DTIDataModule.__init__()` - Lines 149-206
  - Loads drugs.csv, proteins.csv, interactions.csv
  - Splits into 7:2:1 train/val/test
  
- `DTIDataModule._create_fcs_vocabulary()` - Lines 241-305
  - Mines FCS patterns from **training data only**
  - `FCSModule.mine(sequences, max_k=3)` - Line 285
  - Creates pattern vocabulary

**File:** `emm_dti/models/fcs.py`

**Key Classes:**
- `FCSModule` - Lines 14-153
  - `__init__(min_support=0.3)` - Line 21
  - `mine(sequences, max_k=3)` - Lines 56-116
  - `extract_kmers(sequence, k)` - Lines 40-54
  
- `FragmentVocabulary` - Lines 155-224
  - `build_from_fcs(fcs)` - Lines 175-189
  - Maps patterns to indices: D00000 → [14, 42, 7, ...]

---

### **Stage 2: Sequence Tokenization (SMILES/Sequence → Fragment Indices)**

**File:** `emm_dti/data/loaders.py`

**Key Function:**
- `DTIDataset._sequence_to_indices()` - Lines 99-139
  - **Greedy pattern matching** using FCS patterns
  - Converts SMILES/sequence strings to fragment indices
  - Output shape: (batch_size, max_len)
  - Example: "CC(=O)Nc1" → [D14, D42, D7, ...]

---

### **Stage 3: Embedding Layer + Layer Normalization**

**File:** `emm_dti/models/emm_dti_ssm.py`

**Key Lines:**
- `__init__()` - Line 73
  ```python
  self.embedding = nn.Embedding(vocab_size, fcs_embedding_dim, padding_idx=0)
  ```
  - Input: fragment indices (batch, max_len)
  - Output: embeddings (batch, max_len, 128)

- `__init__()` - Line 74
  ```python
  self.embedding_norm = nn.LayerNorm(fcs_embedding_dim)
  ```

- `forward()` - Lines 143-147
  ```python
  drug_emb = self.embedding(drug_indices)      # (batch, drug_len, 128)
  drug_emb = self.embedding_norm(drug_emb)
  protein_emb = self.embedding(protein_indices) # (batch, protein_len, 128)
  protein_emb = self.embedding_norm(protein_emb)
  ```

---

### **Stage 4: Bidirectional Mamba-SSM Processing**

**File:** `emm_dti/models/mamba_ssm.py`

**Key Class:**
- `BidirectionalMambaSSM` - Lines 25-131
  - `__init__()` - Lines 32-108
    ```python
    self.forward_layers = ModuleList([Mamba(...) for i in range(n_layers)])
    self.backward_layers = ModuleList([Mamba(...) for i in range(n_layers)])
    ```
  
  - `forward()` - Lines 111-136
    ```python
    # Forward direction
    fwd = x  # process left→right
    for layer in self.forward_layers:
        fwd = layer(fwd)
    
    # Backward direction (reverse, process, reverse back)
    bwd = torch.flip(x, [1])
    for layer in self.backward_layers:
        bwd = layer(bwd)
    bwd = torch.flip(bwd, [1])
    
    # Concatenate and project
    combined = torch.cat([fwd, bwd], dim=-1)
    output = self.output_proj(combined)
    ```

**In Main Model:**
- `emm_dti/models/emm_dti_ssm.py` - Lines 77-94
  ```python
  self.drug_mamba = BidirectionalMambaSSM(...)
  self.protein_mamba = BidirectionalMambaSSM(...)
  ```

- `forward()` - Lines 150-151
  ```python
  drug_repr = self.drug_mamba(drug_emb)       # (batch, drug_len, 256)
  protein_repr = self.protein_mamba(protein_emb) # (batch, protein_len, 256)
  ```

**Output:** Context-aware representations for each position

---

### **Stage 5: Sequence Pooling (Global Representation)**

**File:** `emm_dti/models/emm_dti_ssm.py`

**Key Lines:**
- `forward()` - Lines 154-155
  ```python
  drug_pool = drug_repr.mean(dim=1)       # (batch, 256)
  protein_pool = protein_repr.mean(dim=1) # (batch, 256)
  ```
  
**Output:** Single vector per drug/protein (mean pooling over sequence dimension)

---

### **Stage 6: Interaction Matrix (Dot Product)**

**File:** `emm_dti/models/emm_dti_ssm.py`

**Key Lines:**
- `forward()` - Lines 158-160
  ```python
  # Compute outer product: drug_repr @ protein_repr^T
  interaction_matrix = torch.bmm(drug_repr, protein_repr.transpose(1, 2))
  # Shape: (batch, drug_len, protein_len)
  
  interaction_matrix = interaction_matrix.unsqueeze(1)
  # Shape: (batch, 1, drug_len, protein_len) - ready for Conv2D
  ```

**Output:** 2D interaction feature map showing alignment scores between drug/protein positions

---

### **Stage 7: CNN Feature Extraction**

**File:** `emm_dti/models/emm_dti_ssm.py`

**Key Lines:**
- `__init__()` - Lines 97-102
  ```python
  self.interaction_cnn = nn.Sequential(
      nn.Conv2d(1, cnn_out_channels=3, kernel_size=3, padding=1),  # Input: 1 channel
      nn.ReLU(),
      nn.AdaptiveMaxPool2d((1, 1))  # Global max pooling
  )
  ```

- `forward()` - Lines 163-164
  ```python
  cnn_features = self.interaction_cnn(interaction_matrix)  # (batch, 3, 1, 1)
  cnn_features = cnn_features.view(batch_size, -1)        # (batch, 3)
  ```

**Output:** 3 feature channels extracted from interaction matrix

---

### **Stage 8: MLP Predictor (Final Classification)**

**File:** `emm_dti/models/emm_dti_ssm.py`

**Key Lines:**
- `__init__()` - Lines 105-112
  ```python
  mlp_input_dim = cnn_out_channels + 2 * mamba_hidden_dim  # 3 + 256 + 256 = 515
  
  self.predictor = nn.Sequential(
      nn.Linear(515, 256), nn.ReLU(), nn.Dropout(0.1),
      nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
      nn.Linear(128, 1)  # Single logit output
  )
  ```

- `forward()` - Lines 167-168
  ```python
  final_features = torch.cat([cnn_features, drug_pool, protein_pool], dim=1)  # (batch, 515)
  prediction = self.predictor(final_features)  # (batch, 1) - LOGIT
  ```

**Output:** DTI prediction logit (not sigmoid - BCEWithLogitsLoss applies it)

---

## 🔄 COMPLETE FORWARD PASS SUMMARY

| Stage | Input | Operation | Output | File |
|-------|-------|-----------|--------|------|
| **Data** | SMILES, Sequence | Load & FCS mine | Fragment indices | `loaders.py` |
| **Tokenize** | Indices | Greedy matching | (batch, seq_len) | `loaders.py` |
| **Embed** | Indices | Embedding + LayerNorm | (batch, seq_len, 128) | `emm_dti_ssm.py:73-74` |
| **Mamba** | Embeddings | BiMamba-SSM | (batch, seq_len, 256) | `mamba_ssm.py` |
| **Pool** | Mamba output | Mean over seq_len | (batch, 256) | `emm_dti_ssm.py:154-155` |
| **Interact** | Drug+Protein repr | Dot product | (batch, 1, d_len, p_len) | `emm_dti_ssm.py:158-160` |
| **CNN** | Interaction matrix | Conv2D + GlobalMaxPool | (batch, 3) | `emm_dti_ssm.py:163-164` |
| **MLP** | CNN + Pool features | 3-layer FC | (batch, 1) logit | `emm_dti_ssm.py:167-168` |

---

## 📋 KEY FILE LOCATIONS

**Core Architecture:**
- `emm_dti/models/emm_dti_ssm.py` - Main model (lines 21-224)
- `emm_dti/models/mamba_ssm.py` - Bidirectional Mamba-SSM
- `emm_dti/models/fcs.py` - FCS mining module

**Data Pipeline:**
- `emm_dti/data/loaders.py` - Data loading, FCS vocabulary, tokenization

**Training & Evaluation:**
- `emm_dti/training/trainer.py` - Training loop (lines 136-350)
- `emm_dti/training/metrics.py` - All evaluation metrics

---

## ✅ READY TO REPLICATE?

You now have the complete file mapping from:
- **Input:** Drug SMILES + Protein Sequence
- **→ Through:** 8 architectural stages  
- **→ To Output:** DTI Prediction

Each stage has exact file and line numbers! 🎯
