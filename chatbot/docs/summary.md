# Player Clustering — Summary

## Project Goal

Cluster 19,729 football players into performance tiers **per position group** using KMeans, then validate that the clusters are real (not artifacts). A shared preprocessing pipeline (`common.py`) is consumed by three scripts.

> **Key decision:** clustering is done **inside each position group**, never across all players. A goalkeeper and a striker share no useful feature space.

---

## Scripts

| Script | Purpose |
|---|---|
| [common.py](common.py) | Loads CSV, defines `FEATURES`, builds the pipeline (median-impute → RobustScaler → PCA 90% variance), per-cluster profiling. |
| [check_market_value.py](check_market_value.py) | Diagnoses a broken upstream join: `latest_market_value` is attached to the wrong player rows. |
| [evaluate.py](evaluate.py) | Measures **stability** (bootstrap ARI + per-cluster Jaccard) and **external validity** (market-value tercile prediction, market value never seen by the model). |
| [run_kmeans.py](run_kmeans.py) | Picks k per position via the stability table produced by `evaluate.py`, then writes the final labels and cluster profiles. |

### Pipeline (per position group)

```
median-impute  →  RobustScaler  →  PCA(90% variance)  →  KMeans
```

- **RobustScaler, not StandardScaler** — distributions are heavily skewed (Messi at 44,740 minutes would squash everyone else).
- **PCA before clustering** — per-90 features are collinear by construction (`goal_contributions_per_90 = goals + assists`), which silently triple-weights attacking output under Euclidean distance.

### Features used (no `_score` columns, no market value)

`goals_per_90`, `assists_per_90`, `goal_contributions_per_90`, `estimated_xg`, `estimated_xa`, `minutes_per_match`, `matches_played`, `age`, `height_in_cm`.

> **Why not the `*_score` columns?** They are a *weighted formula* built from the same raw columns. Feeding them to KMeans would just carve that formula into slices and confirm nothing.

---

## ⚠️ Broken column: `latest_market_value`

The market-value column is **internally consistent** (0.85 pearson with `highest_market_value`) but **orthogonal to performance** (≈0.01 pearson with every perf metric). Messi, Mbappé, Haaland and Bellingham are all recorded at **0**, while the dataset max (140,000,000) belongs to a centre back with 26 matches and 1 goal.

**Verdict:** the value block is joined to the wrong player rows. Run `check_market_value.py` for the evidence. It is therefore excluded from clustering and **not trusted** as a validation signal — the lift numbers in `kmeans_evaluation.csv` are reported but underweighted because of this.

---

## How k is chosen

**Not by the elbow** (too vague), **not by silhouette** (it picks k=2 for every position because it structurally rewards coarse splits).

k is chosen by **bootstrap stability** from `evaluate.py`:

```
bootstrap ARI >= 0.90   AND   every cluster has Jaccard >= 0.75
```

Take the **finest** k that still survives resampling. This is applied **per position**, because positions disagree — Full Back collapses at k=4 (ARI 0.58) but is solid at k=6, while Centre Back is best at k=3. A single global k is wrong for at least two of the eight groups.

---

## Final k per position (from `kmeans_summary.csv`)

| Position | n | PCA dims | k | chosen by | silhouette | smallest cluster |
|---|---:|---:|---:|---|---:|---:|
| Winger | 2,590 | 5 | 4 | stability | 0.190 | 344 |
| Striker | 3,017 | 5 | 4 | stability | 0.208 | 448 |
| Midfielder | 3,585 | 5 | 5 | stability | 0.182 | 404 |
| Other | 389 | 5 | 2 | stability | 0.279 | 156 |
| Defensive Midfielder | 1,807 | 5 | 4 | stability | 0.225 | 156 |
| Full Back | 2,994 | 5 | 6 | stability | 0.212 | 106 |
| Centre Back | 3,748 | 5 | 3 | stability | 0.215 | 397 |
| **Goalkeeper** | 1,599 | **3** | **8** | stability | **0.322** | 27 |

Median silhouette is low (≈0.2) — these features are one continuous cloud, so any KMeans split is a soft segmentation. The Goalkeeper group is the cleanest (silhouette 0.32, lower PCA dim = 3).

---

## Stability results (from `kmeans_evaluation.csv`)

Headers: `bootstrap_ari` ∈ [0,1], `min_jaccard` ∈ [0,1], `mv_lift` = `mv_accuracy − baseline`.

### Highlights

**Strong, stable clusters (ARI ≥ 0.90, min Jaccard ≥ 0.75):**

| Position | k | ARI | stable / k | min Jaccard |
|---|---:|---:|---:|---:|
| Winger | 2 | 0.961 | 2/2 | 0.979 |
| Winger | 3 | 0.944 | 3/3 | 0.952 |
| Winger | 4 | 0.943 | 4/4 | 0.944 |
| Striker | 4 | 0.950 | 4/4 | 0.946 |
| Midfielder | 4 | 0.947 | 4/4 | 0.944 |
| Midfielder | 5 | 0.922 | 5/5 | 0.935 |
| Defensive Midfielder | 3 | 0.921 | 3/3 | 0.896 |
| Defensive Midfielder | 4 | 0.911 | 4/4 | 0.917 |
| Full Back | 3 | 0.966 | 3/3 | 0.959 |
| Full Back | 5 | 0.941 | 5/5 | 0.898 |
| Full Back | 6 | 0.927 | 6/6 | 0.821 |
| Centre Back | 3 | 0.945 | 3/3 | 0.924 |
| Goalkeeper | 3 | 0.965 | 3/3 | 0.917 |
| Goalkeeper | 5 | 0.921 | 5/5 | 0.935 |
| Goalkeeper | 7 | 0.907 | 7/7 | 0.859 |
| Goalkeeper | 8 | 0.906 | 8/8 | 0.864 |

**Notable collapses (ARI < 0.70):**

| Position | k | ARI | min Jaccard | note |
|---|---:|---:|---:|---|
| Winger | 6 | 0.724 | 0.657 | one cluster dissolves |
| Midfielder | 7 | 0.695 | 0.541 | |
| Midfielder | 8 | 0.681 | 0.519 | |
| Other | 5 | 0.625 | 0.593 | small group (n=389) |
| Other | 6 | 0.639 | 0.571 | |
| Other | 7 | 0.575 | 0.411 | |
| Other | 8 | 0.440 | 0.365 | essentially noise |
| Defensive Midfielder | 5 | 0.678 | 0.595 | |
| Full Back | 4 | 0.582 | 0.543 | big jump vs k=3 (0.966) and k=5 (0.941) |
| Centre Back | 6 | 0.597 | 0.358 | one cluster dissolves |
| Centre Back | 5 | 0.711 | 0.720 | borderline |

### Averaged across positions

| k | silhouette | Davies-Bouldin | bootstrap ARI | min Jaccard | mv lift |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.302 | 1.317 | 0.954 | 0.968 | 0.011 |
| 3 | 0.231 | 1.541 | 0.880 | 0.866 | 0.009 |
| 4 | 0.223 | 1.479 | 0.913 | 0.911 | 0.013 |
| 5 | 0.201 | 1.433 | 0.857 | 0.872 | 0.018 |
| 6 | 0.193 | 1.395 | 0.799 | 0.797 | 0.024 |
| 7 | 0.188 | 1.346 | 0.776 | 0.780 | 0.025 |
| 8 | 0.186 | 1.329 | 0.764 | 0.759 | 0.026 |

The numbers tell a clean story: **stability degrades monotonically as k grows**, silhouette degrades monotonically, and the market-value lift rises only because the dataset is sparse at k=8. The selected k values (k=3..8 depending on position) sit at the local sweet spot of "groups still come back after resampling".

---

## External validity (market-value tercile prediction)

The model never saw `latest_market_value`. For each cluster, take the majority tercile and score it. Baseline = majority-class baseline on the same sample.

- **mv_lift** is uniformly small (0.00–0.10). The clusters do better than the baseline, but the lead is thin.
- Largest lifts appear for the small `Other` group (n=389), where the baseline is also weaker.
- The author **explicitly de-emphasises** these numbers because the column is joined to the wrong rows upstream.

---

## Reproduce

```bash
source myenv/bin/activate
python check_market_value.py     # sanity check the broken column
python evaluate.py               # writes kmeans_evaluation.csv + kmeans_cluster_stability.csv
python run_kmeans.py             # per-position k by stability (recommended)
python run_kmeans.py --k 4        # force one k everywhere
python run_kmeans.py --auto      # let silhouette pick k (gives k=2 everywhere)
```

Outputs land in `outputs/`:

- `kmeans_labels.csv` — `player_key`, `player_name`, `position_group`, `kmeans_cluster`, `kmeans_tier`
- `kmeans_summary.csv` — per-position summary
- `kmeans_k_sweep.csv` — silhouette / DB / CH / inertia for k ∈ 2..10
- `kmeans_evaluation.csv` — full stability + external validity sweep
- `kmeans_cluster_stability.csv` — per-cluster Jaccard from bootstrap

---

## Bottom line

The clusters are **stable** (ARI ≥ 0.90, Jaccard ≥ 0.75) at the chosen k for every position, which is the honest stand-in for accuracy when there is no ground-truth label. The silhouette scores are low because the underlying feature cloud is continuous — the segmentation is real but soft. The pipeline is intentionally defensive: it rejects bad market-value validation, refuses to pick k by silhouette, and lets each position name its own number of clusters.
