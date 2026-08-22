# EMM-DTI Paper vs. Implementation - Faithful Reproduction Report

## ✅ COMPONENTS FAITHFULLY REPRODUCED FROM PAPER

| Component | Paper Specifies | Implementation | Status |
|-----------|-----------------|-----------------|--------|
| Architecture Flow | FCS → Embed → Mamba → CNN → MLP | ✓ Implemented | ✓ **MATCH** |
| Mamba Module | Bidirectional Selective SSM | ✓ BidirectionalMambaSSM (2 layers) | ✓ **MATCH** |
| CNN Layer | Conv2D on interaction matrix | ✓ Conv2d(1→3, kernel=3×3) | ✓ **MATCH** |
| Loss Function | Binary Cross-Entropy | ✓ BCEWithLogitsLoss | ✓ **MATCH** |
| Data Splits | 7:2:1 (train:val:test) | ✓ Configured in data_module | ✓ **MATCH** |
| Datasets | BIOSNAP, Human, C.elegans | ✓ Supported via --dataset flag | ✓ **MATCH** |
| Evaluation Metrics | Accuracy, Precision, Recall, AUC, AUPR | ✓ compute_metrics() | ✓ **MATCH** |
| FCS Module | Frequent Continuous Subsequence mining | ✓ FCSModule class implemented | ✓ **MATCH** |
| Embedding Layer | Embedding + LayerNorm | ✓ nn.Embedding + LayerNorm | ✓ **MATCH** |

---

## ⚠️ HYPERPARAMETERS - PAPER UNDERSPECIFIED

| Hyperparameter | Paper | Implementation | Source |
|---|---|---|---|
| **Epochs** | NOT SPECIFIED | **200** | config/train_human_ssm.yaml |
| **Batch Size** | NOT SPECIFIED | **16** | config/train_human_ssm.yaml |
| **Learning Rate** | NOT SPECIFIED | **3e-4 (0.0003)** | config/train_human_ssm.yaml |
| **Optimizer** | NOT SPECIFIED | **Adam (weight_decay=1e-4)** | config/train_human_ssm.yaml |
| **Early Stopping Patience** | NOT SPECIFIED | **30** | config/train_human_ssm.yaml |
| **Gradient Clipping** | NOT SPECIFIED | **1.0** | config/train_human_ssm.yaml |
| **Dropout** | NOT SPECIFIED | **0.1** | config/train_human_ssm.yaml |
| **FCS Min Support** | NOT SPECIFIED | **0.3** | FCSModule default |
| **FCS Max K-mer Size** | NOT SPECIFIED | **3** (1,2,3-mers) | FCSModule default |
| **Embedding Dimension D** | NOT SPECIFIED | **128** | config/train_human_ssm.yaml |
| **Mamba Hidden Dimension** | NOT SPECIFIED | **64** | config/train_human_ssm.yaml |
| **Mamba Layers** | NOT SPECIFIED | **2** | config/train_human_ssm.yaml |
| **CNN Output Channels** | NOT SPECIFIED | **3** | config/train_human_ssm.yaml |

---

## ⚠️ CRITICAL MISSING INFORMATION FROM PAPER

### Training Methodology
- ✗ **Number of CV runs/random seeds** - Paper does NOT specify
  - Implementation: 
    - Enzyme & drugbank: 5-fold CV × 3 seeds (42, 123, 2024) = 15 results per dataset
    - Human, BIOSNAP, C.elegans, BindingDB: 5 independent runs × 5 seeds (42, 123, 2024, 456, 789) = 25 results per dataset
  
- ✗ **K-fold configuration** - Paper does NOT specify
  - Implementation: 5-fold stratified cross-validation

- ✗ **Whether FCS mining uses train data only** - Paper mentions "large, unlabeled databases" but doesn't clarify data leakage prevention
  - Implementation: FCS patterns mined from training data only (proper leakage prevention)

- ✗ **MLP decoder architecture** - Paper shows "fully connected" but doesn't specify layer sizes
  - Implementation: 3-layer MLP (input_dim → 256 → 128 → 1)

- ✗ **CNN kernel stride and padding** - Paper shows Conv2d but doesn't specify these
  - Implementation: kernel_size=3×3, stride=1, padding=1 (preserve dimensions)

- ✗ **Activation functions for CNN and MLP** - Not specified in paper
  - Implementation: ReLU for hidden layers, no activation for output (uses BCEWithLogitsLoss)

---

## 📋 VALIDATION SUMMARY

### Architectural Fidelity: **HIGH ✓**
The core architecture components match the paper exactly:
- Multi-stage pipeline (FCS → Embed → Mamba → CNN → MLP)
- Bidirectional Mamba SSM with 2 layers
- Proper interaction matrix computation via dot product
- FCS-based fragment encoding

### Training Rigor: **MODERATE ⚠️**
The training setup follows best practices but relies on reasonable defaults since the paper underspecifies:
- Stratified 5-fold CV is standard practice for DTI prediction
- 3 independent runs with different seeds ensures robustness
- Early stopping on validation metrics is standard
- Gradient clipping and dropout are standard regularization

### Reproducibility Risk: **MEDIUM ⚠️**
Without the paper specifying key hyperparameters, it's possible that:
1. Different hyperparameter choices could yield different results
2. The exact FCS mining configuration (min_support, max_k) may differ from the original
3. The number of CV runs in the paper's experiments is unknown

---

## 🔍 RECOMMENDATIONS FOR FULL FAITHFULNESS

To achieve **maximum faithfulness**, I recommend:

1. **Check Author's GitHub Repository**
   - Search for EMM-DTI official implementation on GitHub
   - Authors likely published their code with exact hyperparameters

2. **Look for Supplementary Material**
   - Many papers have supplementary PDFs with implementation details
   - Check paper's conference proceedings or journal website

3. **Contact Authors**
   - Reach out to corresponding author for hyperparameter details

4. **Cross-Reference Similar Works**
   - Check if authors published related papers with more details
   - Look at citations for standard configurations

5. **Document Assumptions**
   - If going with current setup, document all hyperparameter choices
   - Note: "These values were chosen as reasonable defaults given paper's underspecification"

---

## 📊 CURRENT IMPLEMENTATION STATUS

| Aspect | Status | Confidence |
|--------|--------|-----------|
| **Model Architecture** | ✓ Faithfully reproduced | **HIGH** |
| **Core Algorithm (FCS+Mamba)** | ✓ Correctly implemented | **HIGH** |
| **Training Loop** | ✓ Follows best practices | **MEDIUM-HIGH** |
| **CV Methodology** | ✓ Proper implementation | **MEDIUM** (paper doesn't specify) |
| **Hyperparameter Values** | ⚠️ Reasonable defaults | **MEDIUM** (paper underspecified) |
| **Overall Reproducibility** | ⚠️ Partial (due to paper gaps) | **MEDIUM** |

---

## CONCLUSION

The `train_cv.py` implementation is **architecturally faithful** to the EMM-DTI paper but **operationally uncertain** due to critical missing hyperparameters in the paper itself. The implementation:

✅ Correctly implements all architectural components  
✅ Uses proper data handling and FCS mining  
✅ Follows ML best practices for training and CV  
⚠️ Makes reasonable assumptions for unspecified hyperparameters  

**Note:** Since the paper does not specify hyperparameters, EMM-DTI hyperparameters have been **standardized to match the 9 existing mamba-dti architectures** for fair comparison:
- Batch size: **16** (consistent across all 10 architectures)
- Learning rate: **3e-4** (consistent across all 10 architectures)
- **5 independent seeds: [42, 123, 2024, 456, 789]** for non-stratified datasets
- **3 CV seeds: [42, 123, 2024]** for enzyme & drugbank (5-fold stratified CV only)
- 5-fold stratified cross-validation for enzyme & drugbank (stratified by compound & protein)
- 5 independent runs for human, biosnap, celegans, bindingdb (no stratification)

This ensures **architectural comparison fairness** while maintaining faithful implementation of the EMM-DTI model.
