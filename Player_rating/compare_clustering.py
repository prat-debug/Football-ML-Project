"""Side-by-side comparison of 4 clustering algorithms on the SAME feature space.

For each model, compute:
    Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score,
    number of clusters, noise points (DBSCAN only), noise %,
    and whether any cluster ended up "tiny" (<5% of dataset).

Writes outputs/clustering_comparison.csv in the requested format.

Usage:
    source myenv/bin/activate
    python compare_clustering.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
)
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture

from common import OUT, build_matrix, load, position_groups

SEED = 42
N_CLUSTERS = 4  # fixed across models for a fair comparison
DBSCAN_EPS = 1.5  # tuned for the PCA(90%) space used by build_matrix
DBSCAN_MIN_SAMPLES = 20
TINY_CLUSTER_FRAC = 0.05  # cluster smaller than 5% of dataset -> warn


def _safe_scores(X, labels):
    """Compute the 3 internal metrics, gracefully handling edge cases."""
    valid = labels != -1
    Xv, lv = X[valid], labels[valid]

    n_clusters = len(set(lv))
    if n_clusters < 2:
        return 0.0, np.inf, 0.0, n_clusters

    sil = silhouette_score(Xv, lv)
    db = davies_bouldin_score(Xv, lv)
    ch = calinski_harabasz_score(Xv, lv)
    return sil, db, ch, n_clusters


def _noise_stats(labels):
    n_total = len(labels)
    n_noise = int((labels == -1).sum())
    pct = 100.0 * n_noise / n_total if n_total else 0.0
    return n_noise, pct


def _tiny_warning(labels, n_total):
    """True if any non-noise cluster is smaller than TINY_CLUSTER_FRAC of n."""
    if n_total == 0:
        return False
    counts = pd.Series(labels[labels != -1]).value_counts()
    if counts.empty:
        return False
    return bool((counts < TINY_CLUSTER_FRAC * n_total).any())


def _aggregate_one_position(X, sub):
    """Run all 4 algorithms on one position group's PCA matrix."""
    n = X.shape[0]
    rows = []

    # --- Hierarchical (Ward) ---
    hc = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward").fit(X)
    sil, db, ch, k = _safe_scores(X, hc.labels_)
    n_noise, pct = _noise_stats(hc.labels_)
    rows.append(
        {
            "Model": "Hierarchical",
            "Silhouette Score": sil,
            "Davies-Bouldin Index": db,
            "Calinski-Harabasz Score": ch,
            "Number of Clusters": k,
            "Noise Points": n_noise,
            "Noise Percentage": round(pct, 4),
            "No Tiny Cluster Warning": not _tiny_warning(hc.labels_, n),
        }
    )

    # --- GMM (full covariance) ---
    gmm = GaussianMixture(
        n_components=N_CLUSTERS,
        covariance_type="full",
        n_init=5,
        random_state=SEED,
    ).fit(X)
    gmm_labels = gmm.predict(X)
    sil, db, ch, k = _safe_scores(X, gmm_labels)
    n_noise, pct = _noise_stats(gmm_labels)
    rows.append(
        {
            "Model": "GMM",
            "Silhouette Score": sil,
            "Davies-Bouldin Index": db,
            "Calinski-Harabasz Score": ch,
            "Number of Clusters": k,
            "Noise Points": n_noise,
            "Noise Percentage": round(pct, 4),
            "No Tiny Cluster Warning": not _tiny_warning(gmm_labels, n),
        }
    )

    # --- KMeans ---
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=SEED).fit(X)
    sil, db, ch, k = _safe_scores(X, km.labels_)
    n_noise, pct = _noise_stats(km.labels_)
    rows.append(
        {
            "Model": "KMeans",
            "Silhouette Score": sil,
            "Davies-Bouldin Index": db,
            "Calinski-Harabasz Score": ch,
            "Number of Clusters": k,
            "Noise Points": n_noise,
            "Noise Percentage": round(pct, 4),
            "No Tiny Cluster Warning": not _tiny_warning(km.labels_, n),
        }
    )

    # --- DBSCAN ---
    db_model = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit(X)
    db_labels = db_model.labels_
    sil, db_score, ch, k = _safe_scores(X, db_labels)
    n_noise, pct = _noise_stats(db_labels)
    rows.append(
        {
            "Model": "DBSCAN",
            "Silhouette Score": sil,
            "Davies-Bouldin Index": db_score,
            "Calinski-Harabasz Score": ch,
            "Number of Clusters": k,
            "Noise Points": n_noise,
            "Noise Percentage": round(pct, 4),
            "No Tiny Cluster Warning": not _tiny_warning(db_labels, n),
        }
    )

    return rows


def main():
    df = load()
    all_rows = []

    for pos, sub in position_groups(df):
        X, _ = build_matrix(sub)
        print(f"\n--- {pos} (n={len(sub)}, pca_dims={X.shape[1]}) ---")
        pos_rows = _aggregate_one_position(X, sub)
        for r in pos_rows:
            r["position_group"] = pos
            all_rows.append(r)
            print(
                f"  {r['Model']:14s}  sil={r['Silhouette Score']:.4f}  "
                f"db={r['Davies-Bouldin Index']:.4f}  ch={r['Calinski-Harabasz Score']:.2f}  "
                f"k={r['Number of Clusters']}  noise={r['Noise Points']} ({r['Noise Percentage']:.2f}%)"
            )
        # mimic the example output: one summary row per model, averaged over positions
        # but the user's example shows a single-table comparison — we'll write BOTH:
        #   clustering_comparison.csv  -> averaged across positions (the headline table)
        #   clustering_comparison_by_position.csv -> per-position detail

    df_long = pd.DataFrame(all_rows)

    # Headline: mean across positions for each model
    headline = (
        df_long.groupby("Model", sort=False)
        .agg(
            {
                "Silhouette Score": "mean",
                "Davies-Bouldin Index": "mean",
                "Calinski-Harabasz Score": "mean",
                "Number of Clusters": "mean",
                "Noise Points": "mean",
                "Noise Percentage": "mean",
                "No Tiny Cluster Warning": "all",
            }
        )
        .reset_index()
    )

    # Order rows like the example
    order = ["Hierarchical", "GMM", "KMeans", "DBSCAN"]
    headline["Model"] = pd.Categorical(headline["Model"], categories=order, ordered=True)
    headline = headline.sort_values("Model").reset_index(drop=True)

    headline.to_csv(OUT / "clustering_comparison.csv", index=False)
    df_long.to_csv(OUT / "clustering_comparison_by_position.csv", index=False)

    print("\n" + "=" * 100)
    print("HEADLINE (averaged across all 8 position groups)")
    print("=" * 100)
    cols = [
        "Model",
        "Silhouette Score",
        "Davies-Bouldin Index",
        "Calinski-Harabasz Score",
        "Number of Clusters",
        "Noise Points",
        "Noise Percentage",
        "No Tiny Cluster Warning",
    ]
    print(headline[cols].to_string(index=False))
    print(f"\nWrote {OUT}/clustering_comparison.csv")
    print(f"Wrote {OUT}/clustering_comparison_by_position.csv")


if __name__ == "__main__":
    main()
