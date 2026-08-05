# Pipeline internals (extracted from source docstrings)

How each script works, in its authors' own words. Regenerated from the
`.py` files, so this never drifts from the code.

## common.py

Shared loading + preprocessing for player clustering.

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

### common.py — position_groups()

Yield (name, sub_dataframe) for every position group big enough to cluster.

### common.py — build_matrix()

Impute (median) -> RobustScaler -> PCA(90% variance).

RobustScaler, not StandardScaler: the distributions are heavily skewed.
Messi at 526 matches / 44,740 minutes would otherwise drag the mean and
squash everyone else into one blob. Median/IQR scaling is immune to that.

PCA before clustering: the per-90 features are collinear by construction
(goal_contributions_per_90 = goals + assists), and both KMeans and DBSCAN
use Euclidean distance -- that collinearity silently triple-weights
attacking output.

### common.py — profile()

Per-cluster profile, ranked best -> worst by goal contributions per 90.

Includes the held-out market value so you can sanity check the ranking.

### common.py — market_value_separation()

Ratio of best-to-worst cluster mean market value, ignoring noise.

This is the external validation number: market value never entered the
clustering, so a high ratio means the groups captured something real.
Returns np.nan when there are fewer than 2 usable clusters.

## check_market_value.py

Evidence that latest_market_value is joined to the wrong player rows.

Run this before trusting any result that uses market value as a target or as a
validation signal.

Usage:
    source myenv/bin/activate
    python check_market_value.py

## evaluate.py

How good is the KMeans clustering, really?

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

### evaluate.py — bootstrap_stability()

Refit on subsamples; measure whether the same clusters reappear.

Returns (mean_ari, per_cluster_jaccard). ARI compares the reference
labelling to the refit one on the shared points. Jaccard tracks each
individual cluster's survival, which ARI averages away.

### evaluate.py — market_value_accuracy()

Can the cluster predict a market-value tercile it never saw?

Restricted to players with a non-zero market value (63.7% of the column is
zero, and that zero mass would otherwise be the answer to everything).
Majority-class rule per cluster, scored against a 33.3% baseline.

## run_kmeans.py

KMeans clustering of players, run separately per position group.

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

### run_kmeans.py — stability_k_table()

Per-position k = the finest split that survives resampling.

Returns {} when evaluate.py has not been run yet, so callers fall back.

### run_kmeans.py — pick_k()

Sweep k and keep the one with the best silhouette score.

## classify_players.py

Classify every player into Elite / Good / Average / Poor.

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

### classify_players.py — cluster_centroids()

Per (position_group, kmeans_cluster) summary stats.

Returned columns are the means of every RANK_FEATURE plus the cluster size,
which is used to break ties (a tiny cluster never has its own category).

### classify_players.py — rank_clusters()

Rank clusters inside each position group, best = 1.

Composite rank score = mean of the per-feature rank (so a cluster that is
top-1 in goals but only top-3 in assists is still ranked above one that is
top-3 in goals and top-1 in assists). Ties broken by cluster size.

### classify_players.py — assign_categories()

Split each position's clusters into Elite / Good / Average / Poor.

Categories are *quartile bands* of the within-position rank ordering:
    top 25%   -> Elite
    next 25%  -> Good
    next 25%  -> Average
    bottom 25%-> Poor

That gives a fixed 4-bucket split regardless of how many clusters the
position actually has. A position with only 2 clusters (the "Other" bucket
in the data) gets Elite + Poor, and the middle two buckets are empty --
which is the honest answer for a group with no in-between structure.

### classify_players.py — build_player_category_map()

(position_group, cluster_id) -> category string.

### classify_players.py — player_category()

Vectorised lookup of category for every player row.
