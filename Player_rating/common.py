"""Shared loading + preprocessing for player clustering.

Design decisions worth knowing before you change anything here:

1. We cluster INSIDE each position_group, never across all 19,729 players.
   A goalkeeper and a striker in the same feature space is meaningless --
   GKs have ~0 goals, so they would just eat a cluster slot.

2. FEATURES contains raw stats only. The *_score columns (talent_score,
   performance_score, physicality_score, ...) are a weighted formula built
   from these same raw columns:
       talent_score      ~= 0.95*performance_score + 0.16*physicality_score
       performance_score ~= f(goal_contribution_score, xg_score, xa_score, ...)
   Feeding them to a clustering algorithm just carves that formula into
   slices and confirms nothing.

3. latest_market_value is EXCLUDED from clustering and is currently UNUSABLE
   as a validation signal too -- see the warning below. Run
   `python check_market_value.py` for the evidence.

   *** BROKEN COLUMN WARNING ***
   latest_market_value correlates with every performance metric at r ~ 0.01,
   i.e. not at all. Messi, Mbappe, Haaland and Bellingham are all recorded at
   0, while the highest value in the dataset (140,000,000) belongs to a centre
   back with 26 matches and 1 goal. It correlates 0.85 with
   highest_market_value, so the market-value block is internally consistent
   but attached to the wrong player rows -- a merge/join bug upstream.
   Do not use it to validate anything until that join is fixed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

CSV = Path(__file__).parent / "clean_player_features.csv"
OUT = Path(__file__).parent / "outputs"

FEATURES = [
    "goals_per_90",
    "assists_per_90",
    "goal_contributions_per_90",
    "estimated_xg",
    "estimated_xa",
    "minutes_per_match",
    "matches_played",
    "age",
    "height_in_cm",
]

VALIDATION = "latest_market_value"
MIN_PLAYERS = 200  # position groups smaller than this are not worth clustering


def load():
    df = pd.read_csv(CSV)
    OUT.mkdir(exist_ok=True)
    return df


def position_groups(df):
    """Yield (name, sub_dataframe) for every position group big enough to cluster."""
    for name, sub in df.groupby("position_group", sort=False):
        if len(sub) >= MIN_PLAYERS:
            yield name, sub.reset_index(drop=True)


def build_matrix(sub, pca_var=0.90, seed=42):
    """Impute (median) -> RobustScaler -> PCA(90% variance).

    RobustScaler, not StandardScaler: the distributions are heavily skewed.
    Messi at 526 matches / 44,740 minutes would otherwise drag the mean and
    squash everyone else into one blob. Median/IQR scaling is immune to that.

    PCA before clustering: the per-90 features are collinear by construction
    (goal_contributions_per_90 = goals + assists), and both KMeans and DBSCAN
    use Euclidean distance -- that collinearity silently triple-weights
    attacking output.
    """
    pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            ("pca", PCA(n_components=pca_var, random_state=seed)),
        ]
    )
    X = pipe.fit_transform(sub[FEATURES])
    return X, pipe


def profile(sub, labels, label_col="cluster"):
    """Per-cluster profile, ranked best -> worst by goal contributions per 90.

    Includes the held-out market value so you can sanity check the ranking.
    """
    p = sub.copy()
    p[label_col] = labels

    agg = (
        p.groupby(label_col)
        .agg(
            n=("player_name", "size"),
            goals_p90=("goals_per_90", "mean"),
            assists_p90=("assists_per_90", "mean"),
            gc_p90=("goal_contributions_per_90", "mean"),
            mins_per_match=("minutes_per_match", "mean"),
            matches=("matches_played", "mean"),
            age=("age", "mean"),
            height=("height_in_cm", "mean"),
            mkt_value_m=(VALIDATION, lambda s: s.mean() / 1e6),
        )
        .round(2)
    )

    # example players = the most-capped members of each cluster
    examples = (
        p.sort_values("matches_played", ascending=False)
        .groupby(label_col)["player_name"]
        .apply(lambda s: ", ".join(s.head(3)))
    )
    agg["examples"] = examples

    # noise (-1) always sorts last; real clusters rank by attacking output
    agg["_rank_key"] = np.where(agg.index == -1, -np.inf, agg["gc_p90"])
    agg = agg.sort_values("_rank_key", ascending=False).drop(columns="_rank_key")
    return agg


def market_value_separation(sub, labels):
    """Ratio of best-to-worst cluster mean market value, ignoring noise.

    This is the external validation number: market value never entered the
    clustering, so a high ratio means the groups captured something real.
    Returns np.nan when there are fewer than 2 usable clusters.
    """
    mask = labels != -1
    if mask.sum() == 0:
        return np.nan
    means = sub.loc[mask, VALIDATION].groupby(labels[mask]).mean()
    if len(means) < 2 or means.max() == 0:
        return np.nan
    lo = means.min()
    return float(means.max() / lo) if lo > 0 else np.inf
