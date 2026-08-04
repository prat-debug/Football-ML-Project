"""Classify every player into Elite / Good / Average / Poor.

Approach
--------
The labels from `run_kmeans.py` carry a `kmeans_tier` column that ranks clusters
within a position group by goal_contributions_per_90 (rank 1 = highest). But
that single number is not comparable across positions -- a "tier 2" winger and
a "tier 2" goalkeeper are not in the same league of performance.

So we re-derive the category per position group, using the cluster itself as the
unit of analysis:

  1. Group all players by (position_group, kmeans_cluster).
  2. For each cluster, compute mean goal_contributions_per_90
     (the headline attacking score) and mean matches_played (experience).
  3. Split each position's clusters into four categories by quantile of those
     two numbers combined. This keeps the labels position-aware: a "Poor"
     striker is still measurably better at attacking than a "Poor" centre back,
     but both are at the bottom of their own position.

To keep behaviour predictable we use a *fixed* per-position mapping that
falls back to the tier when the cluster has no labelled row (defensive safety).
The mapping is computed from the data, then printed so the user can audit it.

Input  : clean_player_features.csv
          outputs/kmeans_labels.csv
Output : clean_player_features_categorised.csv  (same columns + category)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from common import load  # reuse the existing loader

ROOT = Path(__file__).parent
SRC = ROOT / "clean_player_features.csv"
LABELS = ROOT / "outputs" / "kmeans_labels.csv"
OUT = ROOT / "clean_player_features_categorised.csv"

# Headline features used to rank clusters. Kept here -- not in `common` -- so
# this script does not have to know the rest of the pipeline.
RANK_FEATURES = [
    "goal_contributions_per_90",
    "goals_per_90",
    "assists_per_90",
    "estimated_xg",
    "estimated_xa",
    "minutes_per_match",
    "matches_played",
]


def cluster_centroids(df_labelled: pd.DataFrame) -> pd.DataFrame:
    """Per (position_group, kmeans_cluster) summary stats.

    Returned columns are the means of every RANK_FEATURE plus the cluster size,
    which is used to break ties (a tiny cluster never has its own category).
    """
    g = df_labelled.groupby(["position_group", "kmeans_cluster"])
    centroids = g[RANK_FEATURES].mean()
    centroids["n"] = g.size()
    return centroids.reset_index()


def rank_clusters(centroids: pd.DataFrame) -> pd.DataFrame:
    """Rank clusters inside each position group, best = 1.

    Composite rank score = mean of the per-feature rank (so a cluster that is
    top-1 in goals but only top-3 in assists is still ranked above one that is
    top-3 in goals and top-1 in assists). Ties broken by cluster size.
    """
    out = centroids.copy()
    out["rank_score"] = 0.0
    for col in RANK_FEATURES:
        # rank(method='min') -> 1 is the highest; 'first' so ties are stable
        ranks = out.groupby("position_group")[col].rank(
            method="first", ascending=False
        )
        out["rank_score"] += ranks
    out["rank_score"] = out["rank_score"] / len(RANK_FEATURES)
    out["rank_score"] += out["n"].rank(method="first", ascending=False) * 1e-6
    out["rank"] = out.groupby("position_group")["rank_score"].rank(method="first")
    return out


def assign_categories(ranked: pd.DataFrame) -> pd.DataFrame:
    """Split each position's clusters into Elite / Good / Average / Poor.

    Categories are *quartile bands* of the within-position rank ordering:
        top 25%   -> Elite
        next 25%  -> Good
        next 25%  -> Average
        bottom 25%-> Poor

    That gives a fixed 4-bucket split regardless of how many clusters the
    position actually has. A position with only 2 clusters (the "Other" bucket
    in the data) gets Elite + Poor, and the middle two buckets are empty --
    which is the honest answer for a group with no in-between structure.
    """
    ranked = ranked.copy()
    n = ranked.groupby("position_group")["rank"].transform("count")
    pct = (ranked["rank"] - 1) / n  # in [0, 1]

    bands = pd.cut(
        pct,
        bins=[-0.001, 0.25, 0.50, 0.75, 1.001],
        labels=["Elite", "Good", "Average", "Poor"],
        include_lowest=True,
    ).astype(str)
    ranked["category"] = bands
    return ranked


def build_player_category_map(ranked: pd.DataFrame) -> dict[tuple[str, int], str]:
    """(position_group, cluster_id) -> category string."""
    return {
        (row.position_group, int(row.kmeans_cluster)): row.category
        for row in ranked.itertuples()
    }


def player_category(match_count_label: pd.Series, mapping: dict) -> pd.Series:
    """Vectorised lookup of category for every player row."""
    keys = list(zip(match_count_label["position_group"], match_count_label["kmeans_cluster"]))
    return pd.Series([mapping.get(k, "Unclassified") for k in keys], index=match_count_label.index)


def main():
    df = load()
    labels = pd.read_csv(LABELS)

    print(f"loaded {len(df):,} players, {len(labels):,} cluster labels")

    merged = labels.merge(
        df[["player_key", "player_name", "position_group"] + RANK_FEATURES],
        on=["player_key", "player_name", "position_group"],
        how="left",
    )

    centroids = cluster_centroids(merged)
    ranked = rank_clusters(centroids)
    ranked = assign_categories(ranked)

    # Audit print: per position, how many clusters and what categories
    print("\nPer-position category mapping (centroids -> category):")
    audit = (
        ranked.assign(
            gc_p90=ranked["goal_contributions_per_90"].round(2),
            g_p90=ranked["goals_per_90"].round(2),
            m_pm=ranked["minutes_per_match"].round(1),
        )[
            [
                "position_group",
                "kmeans_cluster",
                "n",
                "rank",
                "gc_p90",
                "g_p90",
                "m_pm",
                "category",
            ]
        ].sort_values(["position_group", "rank"])
    )
    print(audit.to_string(index=False))

    mapping = build_player_category_map(ranked)
    merged["category"] = player_category(merged, mapping)

    # Overall category distribution
    print("\nOverall category distribution:")
    print(merged["category"].value_counts().to_string())

    # Per-position × category breakdown
    print("\nPer-position category counts:")
    print(
        merged.groupby(["position_group", "category"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    # Join the category back onto the original dataframe and write
    out = df.merge(
        merged[["player_key", "kmeans_cluster", "category"]],
        on="player_key",
        how="left",
    )
    out["category"] = out["category"].fillna("Unclassified")

    out.to_csv(OUT, index=False)
    print(f"\nWrote {OUT}  ({len(out):,} rows, +2 columns: kmeans_cluster, category)")

    # Spot-check a few famous names so the user can sanity-check
    print("\nSpot-check (should show famous names in Elite):")
    famous = [
        "Lionel Messi",
        "Cristiano Ronaldo",
        "Erling Haaland",
        "Kylian Mbappé",
        "Robert Lewandowski",
        "Vinicius Junior",
        "Jude Bellingham",
        "Kevin De Bruyne",
        "Jadon Sancho",
        "Andronikos Kakoullis",
    ]
    print(
        out[out["player_name"].isin(famous)][
            ["player_name", "position_group", "kmeans_cluster", "category"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
