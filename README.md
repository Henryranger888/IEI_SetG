# IEI Gene Prediction Pipeline 

This repository contains a machine-learning pipeline to prioritize candidate inborn errors of immunity (IEI) genes using network embeddings from a protein–protein interaction (PPI) graph (e.g. STRING with Set2Gaussian embeddings).

The pipeline:

1. Trains and tunes classifiers on **IEI vs non-immune genes** using 5-fold cross-validation (no SMOTE).
2. Selects the best model based on **PR-AUC**.
3. Derives a **global F1-optimal probability threshold** from out-of-fold predictions.
4. Retrains the best model on an updated IEI gene set.
5. Scores all **unlabeled STRING genes**, outputting a ranked list of candidate IEI genes.

---

## File Layout & Inputs

The main script is:

- `iei_pipeline_nosmote.py` – end-to-end training, model selection, thresholding, and prediction.

The script expects the following inputs (paths are defined at the top of the script):

```python
BASE_DIR   = Path("/home/ldap_henryranger/set2/reactome")

EMBED_PATH = BASE_DIR / "output_embedg2g_node_emb.out.npy"      # gene embedding matrix
STRING_PATH = BASE_DIR / "STRING_gene.txt"                      # STRING gene IDs (one per line)
IEI14_PATH  = BASE_DIR / "IEI_2014_filter.txt"                  # positive labels (tuning set)
IEI24_PATH  = BASE_DIR / "IEI_2022_filter.txt"                  # positive labels (final training)
NEG_PATH    = BASE_DIR / "nonimmune_confirm_filter_v2.txt"      # negative labels
OUT_CSV_PATH = BASE_DIR / "pred_unlabeled_ranked.csv"           # output predictions
