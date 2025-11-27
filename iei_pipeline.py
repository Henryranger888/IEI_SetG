#!/usr/bin/env python3
"""
IEI vs non-immune prediction pipeline (no SMOTE)

Workflow
--------
1. Load Set2Gaussian gene embeddings and labeled gene sets:
   - IEI_2014_filter.txt  → positive labels for tuning
   - IEI_2022_filter.txt  → positive labels for final training
   - nonimmune_confirm_filter_v2.txt → negative labels

2. 5-fold cross-validated hyper-parameter search on:
   - IEI_2014 (+) vs non-immune v2 (–)
   - Models (all wrapped as: StandardScaler → classifier):
        - SVM
        - XGBoost
        - Random Forest
        - MLP
   - Scoring metric: PR-AUC (average precision)

3. For each tuned model:
   - Get out-of-fold predicted probabilities via cross_val_predict
   - Compute Acc / Prec / Rec / F1 using an F1-optimised threshold
   - Compute ROC-AUC and PR-AUC

4. Build a soft-voting ensemble (SVM + XGB, weights = PR-AUC of each model)
   - Evaluate with the same metrics
   - Select the best model by PR-AUC among {SVM, XGB, RF, MLP, Vote}

5. Using the best model:
   - Derive a global F1-optimal threshold from out-of-fold probabilities
   - Retrain on IEI_2022 (+) vs non-immune (–)

6. Predict all remaining STRING genes (unseen during training):
   - Output: pred_unlabeled_ranked.csv
     columns: gene, probability, is_pred_IEI
"""

from __future__ import annotations

from pathlib import Path
import random
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline  # used just as a generic pipeline wrapper
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ----------------------------- config ---------------------------------

# Base directory – adjust this if you move the pipeline
BASE_DIR = Path("/home/ldap_henryranger/set2/reactome")

EMBED_PATH   = BASE_DIR / "output_embedg2g_node_emb.out.npy"
STRING_PATH  = BASE_DIR / "STRING_gene.txt"
IEI14_PATH   = BASE_DIR / "IEI_2014_filter.txt"
IEI24_PATH   = BASE_DIR / "IEI_2022_filter.txt"
NEG_PATH     = BASE_DIR / "nonimmune_confirm_filter_v2.txt"
OUT_CSV_PATH = BASE_DIR / "pred_unlabeled_ranked.csv"

N_SPLITS_CV = 5
N_ITER_SEARCH = 25
RANDOM_STATE = 42

warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)


# --------------------------- helper functions -------------------------


def read_list(path: Path) -> List[str]:
    """Read a simple one-column text file (one gene per line, no header)."""
    with path.open() as f:
        return [ln.strip() for ln in f if ln.strip()]


def _average_precision(y_true, y_score, **_) -> float:
    """Robust PR-AUC scorer for RandomizedSearchCV (ignores extra kwargs)."""
    return average_precision_score(y_true, y_score)


PR_SCORER = make_scorer(_average_precision, needs_threshold=True)


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Choose the probability threshold that maximizes F1,
    based on the precision–recall curve.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    if thresholds.size == 0:
        return 0.5  # fallback
    return float(thresholds[np.nanargmax(f1)])


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Compute classification + ranking metrics using an F1-optimised threshold.
    Returns: Acc, Prec, Rec, F1, ROC, PR.
    """
    thr = best_f1_threshold(y_true, y_prob)
    y_pred = (y_prob >= thr).astype(int)

    return {
        "Acc":  accuracy_score(y_true, y_pred),
        "Prec": precision_score(y_true, y_pred, zero_division=0),
        "Rec":  recall_score(y_true, y_pred, zero_division=0),
        "F1":   f1_score(y_true, y_pred, zero_division=0),
        "ROC":  roc_auc_score(y_true, y_prob),
        "PR":   average_precision_score(y_true, y_prob),
    }


def build_pipelines(y: np.ndarray) -> Dict[str, Pipeline]:
    """
    Define model pipelines (StandardScaler → classifier) and
    return a dict of model_name → Pipeline.
    """
    # class ratio for XGBoost
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)

    pipelines = {
        "SVM": Pipeline(
            steps=[
                ("sc", StandardScaler()),
                ("clf", SVC(probability=True, class_weight="balanced")),
            ]
        ),
        "XGB": Pipeline(
            steps=[
                ("sc", StandardScaler()),
                ("clf", XGBClassifier(eval_metric="logloss")),
            ]
        ),
        "RF": Pipeline(
            steps=[
                ("sc", StandardScaler()),
                ("clf", RandomForestClassifier(class_weight="balanced")),
            ]
        ),
        "MLP": Pipeline(
            steps=[
                ("sc", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        max_iter=800,
                        early_stopping=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    # hyperparameter grids (names prefixed with clf__)
    param_grid = {
        "SVM": {
            "clf__C": [0.1, 1, 10, 50],
            "clf__gamma": [0.01, 0.1, 1],
            "clf__kernel": ["rbf"],
        },
        "XGB": {
            "clf__max_depth": [3, 5, 7, 9],
            "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "clf__n_estimators": [300, 500, 700],
            "clf__subsample": [0.6, 0.8, 1.0],
            "clf__colsample_bytree": [0.6, 0.8, 1.0],
            "clf__min_child_weight": [1, 3, 5],
            "clf__gamma": [0, 0.1, 0.2],
            "clf__reg_lambda": [1, 2, 5],
            "clf__reg_alpha": [0, 0.1],
            "clf__scale_pos_weight": [pos_weight],
        },
        "RF": {
            "clf__n_estimators": [400, 600, 800],
            "clf__max_depth": [None, 15, 25],
            "clf__min_samples_leaf": [1, 2],
        },
        "MLP": {
            "clf__hidden_layer_sizes": [(100,), (60, 60)],
            "clf__alpha": [1e-3, 1e-4],
        },
    }

    return pipelines, param_grid


# ------------------------------ main ----------------------------------


def main() -> None:
    # 1. Load embeddings and gene index
    embed = np.load(EMBED_PATH)
    genes = pd.read_csv(STRING_PATH, header=None)[0].str.strip()
    feat_df = pd.DataFrame(embed, index=genes)
    print(f"[INFO] STRING nodes: {len(feat_df):,}")

    # 2. Load labeled gene lists
    pos_2014 = read_list(IEI14_PATH)
    pos_2024 = read_list(IEI24_PATH)
    neg      = read_list(NEG_PATH)

    # 3. Build tuning set (IEI_2014 vs non-immune)
    tune_df = pd.concat(
        [
            pd.DataFrame({"gene": pos_2014, "label": 1}),
            pd.DataFrame({"gene": neg,      "label": 0}),
        ],
        ignore_index=True,
    ).drop_duplicates("gene")

    print(f"[TUNE] +ve={int((tune_df.label == 1).sum())}  "
          f"-ve={int((tune_df.label == 0).sum())}")

    X = feat_df.loc[tune_df.gene]
    y = tune_df.label.values

    cv = StratifiedKFold(
        n_splits=N_SPLITS_CV,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # 4. Build models and parameter grids
    pipelines, param_grid = build_pipelines(y)

    print("\n[CROSS-VALIDATION] 5-fold random search (no SMOTE)")
    metrics: Dict[str, Dict[str, float]] = {}
    tuned_models: Dict[str, Pipeline] = {}

    # 5. Hyper-parameter search per model
    for name, pipe in pipelines.items():
        print(f"\n  > Tuning {name}")
        rs = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grid[name],
            n_iter=N_ITER_SEARCH,
            cv=cv,
            scoring=PR_SCORER,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=0,
        )
        rs.fit(X, y)
        tuned_models[name] = rs.best_estimator_

        # Out-of-fold probabilities
        prob_oof = cross_val_predict(
            rs.best_estimator_,
            X,
            y,
            cv=cv,
            method="predict_proba",
        )[:, 1]

        metrics[name] = compute_metrics(y, prob_oof)
        m = metrics[name]
        print(
            f"    PR={m['PR']:.3f}  ROC={m['ROC']:.3f}  "
            f"F1={m['F1']:.3f}  Acc={m['Acc']:.3f}"
        )

    # 6. Soft-voting ensemble (SVM + XGB) with PR-AUC weights
    vote = VotingClassifier(
        estimators=[("svm", tuned_models["SVM"]), ("xgb", tuned_models["XGB"])],
        voting="soft",
        weights=[metrics["SVM"]["PR"], metrics["XGB"]["PR"]],
    )
    prob_vote_oof = cross_val_predict(
        vote,
        X,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    metrics["Vote"] = compute_metrics(y, prob_vote_oof)
    vote.fit(X, y)
    tuned_models["Vote"] = vote

    mv = metrics["Vote"]
    print(
        f"\n  Vote ensemble:"
        f" PR={mv['PR']:.3f}  ROC={mv['ROC']:.3f}  "
        f"F1={mv['F1']:.3f}  Acc={mv['Acc']:.3f}"
    )

    # 7. Select best model by PR-AUC
    best_name = max(metrics, key=lambda k: metrics[k]["PR"])
    best_pr = metrics[best_name]["PR"]
    print(f"\n[BEST] {best_name} (mean CV PR-AUC = {best_pr:.3f})")

    # 8. Derive global F1-optimal threshold from out-of-fold probabilities
    best_model_for_thr = tuned_models[best_name]
    oof_prob = cross_val_predict(
        best_model_for_thr,
        X,
        y,
        cv=cv,
        method="predict_proba",
    )[:, 1]
    best_thr = best_f1_threshold(y, oof_prob)
    print(f"[THRESH] Optimal F1 threshold = {best_thr:.4f}")

    # 9. Retrain best model on IEI_2022 (+) vs non-immune (–)
    train_df = pd.concat(
        [
            pd.DataFrame({"gene": pos_2024, "label": 1}),
            pd.DataFrame({"gene": neg,      "label": 0}),
        ],
        ignore_index=True,
    ).drop_duplicates("gene")

    print(
        f"\n[TRAIN] +ve={int((train_df.label == 1).sum())}  "
        f"-ve={int((train_df.label == 0).sum())}"
    )

    best_model = tuned_models[best_name]
    best_model.fit(feat_df.loc[train_df.gene], train_df.label.values)

    # 10. Predict all unseen STRING genes
    mask_unseen = ~feat_df.index.isin(train_df.gene)
    n_unseen = int(mask_unseen.sum())
    print(f"[PRED] Unseen STRING nodes: {n_unseen:,}")

    prob_unseen = best_model.predict_proba(feat_df[mask_unseen])[:, 1]
    flag_ieis = (prob_unseen >= best_thr).astype(int)

    out_df = (
        pd.DataFrame(
            {
                "gene": feat_df.index[mask_unseen],
                "probability": prob_unseen,
                "is_pred_IEI": flag_ieis,
            }
        )
        .sort_values("probability", ascending=False)
        .reset_index(drop=True)
    )
    out_df.to_csv(OUT_CSV_PATH, index=False)

    print(f"[DONE] Ranked predictions saved to: {OUT_CSV_PATH}")
    print(f"[INFO] Genes flagged as IEI (prob >= {best_thr:.4f}): {int(flag_ieis.sum())}")


if __name__ == "__main__":
    main()
