#!/usr/bin/env python3
"""
hopkins_ieisets_clustering.py

Compute Hopkins statistics for:
  - IEI 2014 gene set
  - IEI 2022 “new” genes (not in 2014)
  - combined 2014 + 2022 (unique)

And compare each against an empirical null distribution
generated from random Gaussian features.

Outputs:
  - hopkins_statistic_distribution.png   (KDE plot)
"""

from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------
# Paths (adjust BASE / ML_BASE as needed)
# ---------------------------------------------------------------------

BASE    = Path("/home/ldap_henryranger/set2/reactome")
ML_BASE = Path("/home/ldap_henryranger/set2/ML")

EMB_FILE   = BASE / "embedding_feature.npy"
GENE_FILE  = BASE / "sorted_genes_for_embedding.txt"
IEI2014_FILE = ML_BASE / "IUIS_2014.txt"
IEI2022_FILE = BASE / "filtered_IUIS_2022.txt"

OUT_PNG = BASE / "hopkins_statistic_distribution.png"

# Number of Monte Carlo iterations for the null distributions
N_ITER = 10_000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def hopkins_statistic(X: np.ndarray, n: int) -> float:
    """
    Calculate the Hopkins statistic for a dataset X.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data matrix.
    n : int
        Number of random points to use.

    Returns
    -------
    float
        Hopkins statistic in [0, 1].
    """
    # 1) Sample n points from the dataset
    idx = random.sample(range(X.shape[0]), n)
    X_sample = X[idx, :]

    # 2) Generate n random points in the feature space
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    random_points = np.random.uniform(min_vals, max_vals, size=(n, X.shape[1]))

    # 3) Nearest-neighbour distances
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(X)

    # From random points
    u_dist, _ = nn.kneighbors(random_points, n_neighbors=1)
    u_sum = u_dist.sum()

    # From sample points (ignore distance to self)
    w_dist, _ = nn.kneighbors(X_sample, n_neighbors=2)
    w_dist = w_dist[:, 1]
    w_sum = w_dist.sum()

    return float(u_sum / (u_sum + w_sum))


def main() -> None:
    # ---- load feature matrix & gene names --------------------------
    features = np.load(EMB_FILE)   # shape: (n_features, n_nodes) or (n_nodes, n_features)
    if features.shape[0] < features.shape[1]:
        # original code: transpose to (nodes, features)
        features = features.T
    print(f"[INFO] feature matrix shape: {features.shape}")

    with GENE_FILE.open() as f:
        gene_names = [line.strip() for line in f if line.strip()]

    if len(gene_names) != features.shape[0]:
        print(
            "[WARN] Number of genes does not match number of rows in feature "
            f"matrix: {len(gene_names)} vs {features.shape[0]}"
        )

    gene_to_index = {g: i for i, g in enumerate(gene_names)}

    # ---- load IEI gene lists ---------------------------------------
    with IEI2014_FILE.open() as f:
        iuis_2014_nodes = [line.strip() for line in f if line.strip()]
    with IEI2022_FILE.open() as f:
        iuis_2022_nodes = [line.strip() for line in f if line.strip()]

    # 2022 unique = 2022 minus 2014
    iuis_2022_unique_nodes = [g for g in iuis_2022_nodes if g not in iuis_2014_nodes]

    # map to indices, dropping genes not in embedding
    idx_2014 = [gene_to_index[g] for g in iuis_2014_nodes if g in gene_to_index]
    idx_2022 = [gene_to_index[g] for g in iuis_2022_unique_nodes if g in gene_to_index]
    idx_combined = idx_2014 + idx_2022

    features_2014 = features[idx_2014, :]
    features_2022 = features[idx_2022, :]
    features_combined = features[idx_combined, :]

    print(f"[INFO] n_2014 = {features_2014.shape[0]}, "
          f"n_2022_unique = {features_2022.shape[0]}, "
          f"n_combined = {features_combined.shape[0]}")

    # n for Hopkins (cannot exceed smallest group size or 100)
    n_samples = min(100, features_2014.shape[0],
                    features_2022.shape[0], features_combined.shape[0])
    if n_samples < 5:
        raise ValueError("Too few points for Hopkins statistic (n_samples < 5).")

    # ---- 1) observed Hopkins values --------------------------------
    h_true_2014 = hopkins_statistic(features_2014, n=n_samples)
    h_true_2022 = hopkins_statistic(features_2022, n=n_samples)
    h_true_combined = hopkins_statistic(features_combined, n=n_samples)

    print(f"Hopkins (2014):     {h_true_2014:.4f}")
    print(f"Hopkins (2022 new): {h_true_2022:.4f}")
    print(f"Hopkins (combined): {h_true_combined:.4f}")

    # ---- 2) null distributions -------------------------------------
    h_null_2014 = []
    h_null_2022 = []
    h_null_combined = []

    print(f"[INFO] Building null distributions with {N_ITER} iterations...")

    for i in range(N_ITER):
        # random Gaussian reference matrix with same shape as original
        random_features = np.random.randn(*features.shape)

        # sample index sets of same sizes
        rnd_idx_2014 = np.random.choice(random_features.shape[0],
                                        size=len(idx_2014), replace=False)
        rnd_idx_2022 = np.random.choice(random_features.shape[0],
                                        size=len(idx_2022), replace=False)
        rnd_idx_comb = np.random.choice(random_features.shape[0],
                                        size=len(idx_combined), replace=False)

        h_null_2014.append(
            hopkins_statistic(random_features[rnd_idx_2014, :], n=n_samples)
        )
        h_null_2022.append(
            hopkins_statistic(random_features[rnd_idx_2022, :], n=n_samples)
        )
        h_null_combined.append(
            hopkins_statistic(random_features[rnd_idx_comb, :], n=n_samples)
        )

        if (i + 1) % 100 == 0:
            print(f"  iteration {i + 1}/{N_ITER} complete")

    h_null_2014 = np.array(h_null_2014)
    h_null_2022 = np.array(h_null_2022)
    h_null_combined = np.array(h_null_combined)

    # ---- 3) p-values -----------------------------------------------
    p_2014 = np.mean(h_null_2014 >= h_true_2014)
    p_2022 = np.mean(h_null_2022 >= h_true_2022)
    p_comb = np.mean(h_null_combined >= h_true_combined)

    print(f"P-value (2014):     {p_2014:.4f}")
    print(f"P-value (2022 new): {p_2022:.4f}")
    print(f"P-value (combined): {p_comb:.4f}")

    # ---- 4) KDE plots ----------------------------------------------
    plt.figure(figsize=(10, 6))

    sns.kdeplot(h_null_2014, color="blue", label="h_null_2014 KDE")
    plt.axvline(
        h_true_2014,
        color="blue",
        linestyle="dashed",
        linewidth=2,
        label=f"h_true_2014: {h_true_2014:.4f}",
    )

    sns.kdeplot(h_null_2022, color="green", label="h_null_2022 KDE")
    plt.axvline(
        h_true_2022,
        color="green",
        linestyle="dashed",
        linewidth=2,
        label=f"h_true_2022: {h_true_2022:.4f}",
    )

    sns.kdeplot(h_null_combined, color="red", label="h_null_combined KDE")
    plt.axvline(
        h_true_combined,
        color="red",
        linestyle="dashed",
        linewidth=2,
        label=f"h_true_combined: {h_true_combined:.4f}",
    )

    plt.title(
        "Hopkins Statistic Null Distributions for IEI 2014, 2022 (new), and Combined"
    )
    plt.xlabel("Hopkins statistic")
    plt.ylabel("Density")
    plt.legend(loc="upper right")
    plt.tight_layout()

    plt.savefig(OUT_PNG, dpi=300)
    print(f"[OUT] Hopkins KDE figure written to: {OUT_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
