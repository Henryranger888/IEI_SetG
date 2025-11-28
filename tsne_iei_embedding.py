#!/usr/bin/env python3
"""
tsne_iei_embedding.py

Visualise STRING network embeddings with t-SNE, colouring
proteins by IEI / non-immune / unlabelled categories.

Outputs:
  - <BASE>/tsne_embedding_IEI.png
"""

import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

# ---------------------------------------------------------------------
# Paths (adjust BASE if needed)
# ---------------------------------------------------------------------

BASE = Path("/home/ldap_henryranger/set2/reactome")

EMB_FILE = BASE / "output_embedg2g_node_emb.out.npy"
STRING_FILE = BASE / "STRING_gene.txt"

IEI14_FILE = BASE / "IEI_2014_filter.txt"
IEI22_FILE = BASE / "IEI_2022_filter.txt"
IEI24_FILE = BASE / "IEI_2024_filter.txt"
NEG_FILE   = BASE / "nonimmune_confirm_filter_v2.txt"


def read_list(path: Path) -> set[str]:
    """Read a plain-text file (one gene per line) into a set."""
    with path.open() as f:
        return {line.strip() for line in f if line.strip()}


def main() -> None:
    # ---- load embeddings & gene IDs --------------------------------
    emb = np.load(EMB_FILE)  # e.g. shape (n_genes, n_features)
    genes = pd.read_csv(STRING_FILE, header=None)[0].str.strip()
    feat = pd.DataFrame(emb, index=genes)

    print(f"[INFO] embedding matrix: {feat.shape}")

    # ---- load IEI and non-immune sets ------------------------------
    s14 = read_list(IEI14_FILE)
    s22 = read_list(IEI22_FILE)
    s24 = read_list(IEI24_FILE)
    neg = read_list(NEG_FILE)

    # “New” IEI genes first appearing in 2022 / 2024
    new22 = s22 - s14
    new24 = s24 - s22

    # Drop any overlap between negatives and IEI lists (just in case)
    overlap = neg & (s14 | new22 | new24)
    if overlap:
        print(
            f"[WARN] {len(overlap)} genes removed from non-immune negatives "
            "because they appear in an IEI set:"
        )
        print("  " + ", ".join(sorted(overlap)))
        neg -= overlap

    # ---- assign categories -----------------------------------------
    def label(gene: str) -> str:
        if gene in s14:
            return "IEI-2014"
        elif gene in new22:
            return "IEI-2022 (new)"
        elif gene in new24:
            return "IEI-2024 (new)"
        elif gene in neg:
            return "Non-immune"
        else:
            return "Unlabelled"

    feat["category"] = genes.map(label).values
    print(feat["category"].value_counts(dropna=False))

    # ---- t-SNE -----------------------------------------------------
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        random_state=42,
        learning_rate="auto",
    )
    xy = tsne.fit_transform(feat.iloc[:, :500])
    feat["TSNE1"] = xy[:, 0]
    feat["TSNE2"] = xy[:, 1]

    # ---- plot ------------------------------------------------------
    colour_map = {
        "IEI-2014":        "#E24A33",  # orange-red
        "IEI-2022 (new)":  "#348ABD",  # blue
        "IEI-2024 (new)":  "#988ED5",  # purple
        "Non-immune":      "#006400",  # dark green
        "Unlabelled":      "#BDBDBD",  # light grey
    }

    plt.figure(figsize=(9, 7))

    for cat, sub in feat.groupby("category", sort=False):
        if sub.empty:
            continue
        is_iei = cat.startswith("IEI")
        plt.scatter(
            sub["TSNE1"],
            sub["TSNE2"],
            s=12 if is_iei else 6,
            c=colour_map.get(cat, "#000000"),
            alpha=0.85 if is_iei else 0.25,
            edgecolors="none",
            label=cat,
        )

    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.title("t-SNE projection of STRING embeddings by IEI category")
    plt.legend(frameon=False, fontsize=9, markerscale=1.4)
    plt.tight_layout()

    out_png = BASE / "tsne_embedding_IEI.png"
    plt.savefig(out_png, dpi=600)
    print(f"[OUT] t-SNE figure written to: {out_png}")


if __name__ == "__main__":
    main()
