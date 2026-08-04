"""How good is the KMeans clustering, really?

There is no "accuracy" for clustering in the supervised sense -- nobody labelled
these 19,729 players with a true group, so there is nothing to be right or wrong
about. What CAN be measured, and what this script reports:

  1. STABILITY (the honest stand-in for accuracy)
     Refit the clustering on bootstrap resamples and check whether the same
     groups come back. Reported two ways:
       - bootstrap ARI: agreement between the full-data labels and the refit
         labels, corrected for chance. This is the headline number.
       - per-cluster Jaccard (Hennig's criterion): >0.75 stable,
         0.60-0.75 a real but fuzzy pattern, <0.60 noise.
     A cluster that does not survive resampling is an artifact, full stop.

  2. EXTERNAL VALIDITY (accuracy against something never shown to the model)
     latest_market_value is excluded from clustering. So: can cluster
     membership predict a player's market-value tercile better than chance?
     Assign each cluster its majority tercile and score it. Baseline is 33.3%.
     Anything meaningfully above that is real signal, because the algorithm
     never saw the column.

  3. INTERNAL GEOMETRY
     Silhouette, Davies-Bouldin, Calinski-Harabasz. Useful for comparing k,
     useless as an absolute quality claim.

Usage:
    source myenv/bin/activate
    python evaluate.py
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from common import OUT, VALIDATION, build_matrix, load, position_groups

SEED = 42
K_RANGE = range(2, 9)
N_BOOTSTRAP = 30
SUBSAMPLE = 0.8


def bootstrap_stability(X, k, n_boot=N_BOOTSTRAP, frac=SUBSAMPLE, seed=SEED):
    """Refit on subsamples; measure whether the same clusters reappear.

    Returns (mean_ari, per_cluster_jaccard). ARI compares the reference
    labelling to the refit one on the shared points. Jaccard tracks each
    individual cluster's survival, which ARI averages away.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    ref = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    ref_labels = ref.labels_

    aris = []
    jaccard = {c: [] for c in range(k)}

    for b in range(n_boot):
        idx = rng.choice(n, size=int(frac * n), replace=False)
        Xs = X[idx]
        new_labels = KMeans(n_clusters=k, n_init=10, random_state=seed + b + 1).fit_predict(Xs)
        aris.append(adjusted_rand_score(ref_labels[idx], new_labels))

        # for each reference cluster, best-matching new cluster by Jaccard
        for c in range(k):
            a = set(np.flatnonzero(ref_labels[idx] == c))
            if not a:
                continue
            best = 0.0
            for d in range(k):
                bset = set(np.flatnonzero(new_labels == d))
                union = len(a | bset)
                if union:
                    best = max(best, len(a & bset) / union)
            jaccard[c].append(best)

    mean_jac = {c: float(np.mean(v)) if v else np.nan for c, v in jaccard.items()}
    return float(np.mean(aris)), mean_jac


def market_value_accuracy(sub, labels):
    """Can the cluster predict a market-value tercile it never saw?

    Restricted to players with a non-zero market value (63.7% of the column is
    zero, and that zero mass would otherwise be the answer to everything).
    Majority-class rule per cluster, scored against a 33.3% baseline.
    """
    mv = sub[VALIDATION].to_numpy()
    mask = mv > 0
    if mask.sum() < 30:
        return np.nan, np.nan

    tiers = pd.qcut(mv[mask], 3, labels=[0, 1, 2], duplicates="drop").astype(int)
    lab = labels[mask]

    correct = 0
    for c in np.unique(lab):
        in_c = lab == c
        if in_c.sum() == 0:
            continue
        correct += np.bincount(tiers[in_c]).max()
    acc = correct / len(tiers)
    baseline = float(np.bincount(tiers).max() / len(tiers))  # majority-class baseline
    return float(acc), baseline


def main():
    df = load()
    rows, cluster_rows = [], []

    for pos, sub in position_groups(df):
        X, _ = build_matrix(sub)

        for k in K_RANGE:
            km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X)
            labels = km.labels_
            ari, jac = bootstrap_stability(X, k)
            acc, base = market_value_accuracy(sub, labels)

            rows.append(
                {
                    "position_group": pos,
                    "n": len(sub),
                    "k": k,
                    "silhouette": round(float(silhouette_score(X, labels)), 3),
                    "davies_bouldin": round(float(davies_bouldin_score(X, labels)), 3),
                    "calinski_harabasz": round(float(calinski_harabasz_score(X, labels)), 0),
                    "bootstrap_ari": round(ari, 3),
                    "stable_clusters": sum(1 for v in jac.values() if v >= 0.75),
                    "min_jaccard": round(min(jac.values()), 3),
                    "mv_accuracy": round(acc, 3) if not np.isnan(acc) else np.nan,
                    "mv_baseline": round(base, 3) if not np.isnan(base) else np.nan,
                    "mv_lift": round(acc - base, 3) if not np.isnan(acc) else np.nan,
                }
            )
            for c, v in jac.items():
                cluster_rows.append(
                    {"position_group": pos, "k": k, "cluster": c, "jaccard": round(v, 3)}
                )

        print(f"  done: {pos}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "kmeans_evaluation.csv", index=False)
    pd.DataFrame(cluster_rows).to_csv(OUT / "kmeans_cluster_stability.csv", index=False)

    print(f"\n{'#' * 100}\nFULL EVALUATION (every position x every k)\n{'#' * 100}")
    print(res.to_string(index=False))

    print(f"\n{'#' * 100}\nAVERAGED ACROSS POSITIONS -- how does k trade off?\n{'#' * 100}")
    agg = (
        res.groupby("k")[
            ["silhouette", "davies_bouldin", "bootstrap_ari", "min_jaccard", "mv_accuracy", "mv_lift"]
        ]
        .mean()
        .round(3)
    )
    agg["stable_clusters"] = res.groupby("k")["stable_clusters"].mean().round(2)
    print(agg.to_string())

    print(f"\nWrote {OUT}/kmeans_evaluation.csv, kmeans_cluster_stability.csv")


if __name__ == "__main__":
    main()
