# IEI Network Embedding and Gene Prioritisation

This repository contains code to:

1. Train and evaluate machine-learning models to prioritise inborn errors of immunity (IEI) genes from a STRING-based network embedding.
2. Visualise the embedding with t-SNE, colouring nodes by IEI / non-immune / unlabelled categories.
3. Quantify clustering of IEI sets in the embedding using the Hopkins statistic and an empirical null distribution.

The overall goal is to combine network-derived embeddings, curated IEI / non-immune gene sets, and supervised learning to generate a ranked list of candidate IEI genes for follow-up.

> **Note:** All paths under `/home/ldap_henryranger/...` are easily configurable at the top of each script.

---

## Repository Structure

- `iei_pipeline_nosmote.py`  
  Main IEI gene prioritisation pipeline.

- `tsne_iei_embedding.py`  
  t-SNE visualisation of the network embedding, coloured by IEI category.

- `hopkins_ieisets_clustering.py`  
  Hopkins statistic analysis for IEI 2014 / 2022 / combined sets.

- `output_embedg2g_node_emb.out.npy`  
  Network embedding matrix (STRING genes × features).

- `STRING_gene.txt`  
  STRING gene identifiers (one per line), aligned with the embedding.

- `IEI_2014_filter.txt`  
  IEI positives used for model tuning (2014 catalogue).

- `IEI_2022_filter.txt`  
  IEI positives used for final training (2022 catalogue).

- `IEI_2024_filter.txt`  
  IEI positives first appearing in 2024 (for visualisation).

- `nonimmune_confirm_filter_v2.txt`  
  Curated non-immune negative gene list.

- `embedding_feature.npy`, `sorted_genes_for_embedding.txt`  
  Embedding and gene list used for the Hopkins statistic analysis.

- `IUIS_2014.txt`, `filtered_IUIS_2022.txt`  
  Additional IEI lists for the Hopkins analysis.

File names for the input lists match the code; adjust them if your filenames differ.

---

## 1. IEI Gene Prioritisation Pipeline

**Script:** `iei_pipeline_nosmote.py`  

This script trains several classifiers on **IEI-2014 (+)** vs **non-immune v2 (–)** and uses cross-validated PR-AUC to select the best model. It then transfers the model to **IEI-2022 (+)** vs **non-immune v2 (–)**, retrains, and scores all remaining STRING genes.

### 1.1. What the script does

1. **Load data**
   - Embedding: `output_embedg2g_node_emb.out.npy`
   - Gene IDs: `STRING_gene.txt`
   - Positives: `IEI_2014_filter.txt` and `IEI_2022_filter.txt`
   - Negatives: `nonimmune_confirm_filter_v2.txt`

2. **Model tuning on IEI-2014 vs non-immune**
   - Models:
     - SVM (`SVC`, probability enabled, `class_weight='balanced'`)
     - XGBoost (`XGBClassifier`)
     - Random Forest (`RandomForestClassifier`, `class_weight='balanced'`)
     - MLP (`MLPClassifier` with early stopping)
   - 5-fold stratified CV with `RandomizedSearchCV`
   - Custom PR-AUC scorer
   - No SMOTE or resampling; balancing is handled via class weights and thresholding.

3. **Model selection**
   - For each tuned model:
     - Get out-of-fold predicted probabilities (5-fold).
     - Compute accuracy, precision, recall, F1, ROC-AUC, and PR-AUC.
   - Select the best model by **PR-AUC**.
   - Derive a global **F1-optimal probability threshold** from these out-of-fold predictions.

4. **Final training and prediction**
   - Retrain the best model on **IEI-2022 (+)** vs **non-immune (–)**.
   - Apply to **all unseen STRING genes** (not used in final training).
   - Output a ranked list of candidate IEI genes.

### 1.2. Paths and expected input format

At the top of `iei_pipeline_nosmote.py`:

```python
from pathlib import Path

BASE   = Path("/home/ldap_henryranger/set2/reactome")
EMB    = BASE / "output_embedg2g_node_emb.out.npy"
STRING = BASE / "STRING_gene.txt"
IEI14  = BASE / "IEI_2014_filter.txt"
IEI24  = BASE / "IEI_2022_filter.txt"
NEG    = BASE / "nonimmune_confirm_filter_v2.txt"
OUTCSV = BASE / "pred_unlabeled_ranked.csv"
