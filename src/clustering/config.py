"""
config.py
----------

Central configuration file for the Clustering Module.

All paths, feature names and global parameters are defined here
to avoid hardcoding values across multiple files.
"""

from pathlib import Path

# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ==========================================================
# DATA PATHS
# ==========================================================

DATA_DIR = PROJECT_ROOT / "data"

DATASET_PATH = DATA_DIR / "cleaned" / "clean_player_features.csv"

# ==========================================================
# OUTPUT DIRECTORIES
# ==========================================================

MODELS_DIR = PROJECT_ROOT / "models" / "clustering"

RESULTS_DIR = PROJECT_ROOT / "results" / "clustering"

PLOTS_DIR = RESULTS_DIR / "plots"

LINEAR_RESULTS_DIR = RESULTS_DIR / "linear_pca"

KERNEL_RESULTS_DIR = RESULTS_DIR / "kernel_pca"

# ==========================================================
# CREATE DIRECTORIES IF NOT EXISTS
# ==========================================================

MODELS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

LINEAR_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

KERNEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# FEATURES USED FOR CLUSTERING
# ==========================================================

CLUSTER_FEATURES = [

    "age",

    "height_in_cm",

    "matches_played",

    "minutes_per_match",

    "goals_per_90",

    "assists_per_90",

    "estimated_xg",

    "estimated_xa",

    "physicality_score"

]

# Only for cluster interpretation

POSITION_COLUMN = "position_group"

# ==========================================================
# RANDOM STATE
# ==========================================================

RANDOM_STATE = 42

# ==========================================================
# PCA SETTINGS
# ==========================================================

MIN_COMPONENTS = 2

MAX_COMPONENTS = 9

# ==========================================================
# KERNELS TO TEST
# ==========================================================

KERNELS = [

    "rbf",

    "poly",

    "sigmoid",

    "cosine"

]
# ==========================================================
# PCA SETTINGS
# ==========================================================

PCA_COMPONENTS = 6