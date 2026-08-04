"""KMeans clustering of players, run separately per position group.

HOW k IS CHOSEN (default mode: stability)

Not by the elbow -- too vague to defend when someone asks why you picked 5.
Not by silhouette either: it picks k=2 for every single position, because these
features are one continuous cloud and silhouette structurally rewards coarse
splits. k=2 gives a trivial "played a lot / played a little" cut.

Instead k comes from bootstrap stability, computed by evaluate.py: refit the
clustering on 30 resamples and keep the FINEST k that still reproduces. The rule
is the largest k where

    bootstrap ARI >= 0.90   and   every cluster has Jaccard >= 0.75

i.e. the groups come back after resampling and no individual cluster dissolves.
That is applied per position, because the positions do not agree -- Full Back
collapses at k=4 (ARI 0.58) but is solid at k=5 and k=6, while Centre Back is
best at k=3. A single global k is wrong for at least two of the eight groups.

Requires outputs/kmeans_evaluation.csv. Run evaluate.py first; without it this
falls back to a flat k=4.

Usage:
    source myenv/bin/activate
    python evaluate.py              # produces the stability table (~90s)
    python run_kmeans.py            # per-position k by stability (recommended)
    python run_kmeans.py --k 4      # force one k everywhere
    python run_kmeans.py --auto     # let silhouette pick (gives k=2 everywhere)
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from common import OUT, build_matrix, load, position_groups, profile

K_RANGE = range(2, 11)
SEED = 42
DEFAULT_K = 4  # fallback only, when the stability table is missing

STABILITY_FILE = OUT / "kmeans_evaluation.csv"
MIN_ARI = 0.90  # grouping must reproduce across resamples
MIN_JACCARD = 0.75  # Hennig's threshold for "this cluster is real"


def stability_k_table():
    """Per-position k = the finest split that survives resampling.

    Returns {} when evaluate.py has not been run yet, so callers fall back.
    """
    if not STABILITY_FILE.exists():
        return {}

    ev = pd.read_csv(STABILITY_FILE)
    ok = ev[
        (ev["bootstrap_ari"] >= MIN_ARI)
        & (ev["min_jaccard"] >= MIN_JACCARD)
        & (ev["stable_clusters"] == ev["k"])  # no cluster left behind
    ]
    # finest surviving k per position
    return ok.loc[ok.groupby("position_group")["k"].idxmax()].set_index("position_group")["k"].to_dict()


def pick_k(X, k_range=K_RANGE, seed=SEED):
    """Sweep k and keep the one with the best silhouette score."""
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
        rows.append(
            {
                "k": k,
                "silhouette": silhouette_score(X, km.labels_),
                "davies_bouldin": davies_bouldin_score(X, km.labels_),
                "calinski_harabasz": calinski_harabasz_score(X, km.labels_),
                "inertia": km.inertia_,
            }
        )
    sweep = pd.DataFrame(rows)
    best_k = int(sweep.loc[sweep["silhouette"].idxmax(), "k"])
    return best_k, sweep


def main(force_k=None, auto=False):
    df = load()
    labelled, summary, sweeps = [], [], []

    stab = {} if (auto or force_k) else stability_k_table()
    if not (auto or force_k) and not stab:
        print(f"! {STABILITY_FILE.name} not found -- run evaluate.py. Falling back to k={DEFAULT_K}.\n")

    for pos, sub in position_groups(df):
        X, _ = build_matrix(sub)
        silhouette_k, sweep = pick_k(X)

        if auto:
            best_k, why = silhouette_k, "silhouette"
        elif force_k:
            best_k, why = force_k, "forced"
        elif pos in stab:
            best_k, why = int(stab[pos]), "stability"
        else:
            best_k, why = DEFAULT_K, "fallback"
        sweep.insert(0, "position_group", pos)
        sweeps.append(sweep)

        km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(X)
        labels = km.labels_

        prof = profile(sub, labels)
        # rank 1 = strongest cluster in this position (profile() is pre-sorted)
        tier = {c: i + 1 for i, c in enumerate(prof.index)}

        out = sub[["player_key", "player_name", "position_group"]].copy()
        out["kmeans_cluster"] = labels
        out["kmeans_tier"] = [tier[c] for c in labels]
        labelled.append(out)

        # forced k may fall outside K_RANGE, so fall back to computing it
        hit = sweep.loc[sweep["k"] == best_k, "silhouette"]
        sil = float(hit.iloc[0]) if len(hit) else float(silhouette_score(X, labels))
        summary.append(
            {
                "position_group": pos,
                "n": len(sub),
                "pca_dims": X.shape[1],
                "k": best_k,
                "k_chosen_by": why,
                "silhouette": round(sil, 3),
                "smallest_cluster": int(prof["n"].min()),
                # NOTE: no market-value validation column here on purpose --
                # that column is joined to the wrong rows. See
                # check_market_value.py.
            }
        )

        print(
            f"\n{'=' * 78}\n{pos}  (n={len(sub)}, k={best_k} via {why}, "
            f"silhouette={sil:.3f})\n{'=' * 78}"
        )
        print(prof.to_string())

    pd.concat(labelled).to_csv(OUT / "kmeans_labels.csv", index=False)
    pd.concat(sweeps).to_csv(OUT / "kmeans_k_sweep.csv", index=False)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT / "kmeans_summary.csv", index=False)

    print(f"\n{'#' * 78}\nKMEANS SUMMARY\n{'#' * 78}")
    print(summary_df.to_string(index=False))
    print(f"\nWrote {OUT}/kmeans_labels.csv, kmeans_k_sweep.csv, kmeans_summary.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None, help="force one k for every position")
    ap.add_argument("--auto", action="store_true", help="let silhouette pick k per position")
    a = ap.parse_args()
    main(force_k=a.k, auto=a.auto)
