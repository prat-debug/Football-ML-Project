"""
build_final_pipeline.py
-----------------------

Builds the final deployable football clustering pipeline.

Pipeline:
Raw CLUSTER_FEATURES
    -> StandardScaler
    -> RBF Kernel PCA
    -> Final tuned KMeans
    -> Cluster name

Final KMeans parameters were selected during previous tuning.
"""

import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.cluster import KMeans

from config import (
    DATASET_PATH,
    MODELS_DIR,
    CLUSTER_FEATURES,
    PCA_COMPONENTS,
    RANDOM_STATE
)


# ==========================================================
# FINAL SETTINGS
# ==========================================================

CLUSTER_MAPPING = {
    0: "Experienced Defenders",
    1: "Elite Forwards",
    2: "Young Talents",
    3: "Defensive Specialists"
}


FINAL_KMEANS_PARAMS = {
    "n_clusters": 4,
    "init": "k-means++",
    "n_init": 20,
    "max_iter": 300,
    "algorithm": "lloyd",
    "random_state": RANDOM_STATE
}


# ==========================================================
# BUILD PIPELINE
# ==========================================================

def build_final_pipeline():

    print("\n" + "=" * 70)
    print("BUILDING FINAL DEPLOYABLE CLUSTERING PIPELINE")
    print("=" * 70)

    # ------------------------------------------------------
    # 1. Load cleaned dataset
    # ------------------------------------------------------

    print("\nLoading cleaned dataset...")

    df = pd.read_csv(DATASET_PATH)

    missing_columns = [
        column
        for column in CLUSTER_FEATURES
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing clustering features: {missing_columns}"
        )

    X = df[CLUSTER_FEATURES].copy()

    if X.isnull().sum().sum() > 0:
        raise ValueError(
            "Missing values found in clustering features."
        )

    print(f"Players  : {len(X)}")
    print(f"Features : {len(CLUSTER_FEATURES)}")

    # ------------------------------------------------------
    # 2. Fit scaler
    # ------------------------------------------------------

    print("\nFitting StandardScaler...")

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    print(
        f"Scaled Shape : {X_scaled.shape}"
    )

    # ------------------------------------------------------
    # 3. Fit final RBF Kernel PCA
    # ------------------------------------------------------

    print("\nFitting final RBF Kernel PCA...")

    rbf_kpca = KernelPCA(
        n_components=PCA_COMPONENTS,
        kernel="rbf",
        fit_inverse_transform=False,
        random_state=RANDOM_STATE
    )

    X_rbf = rbf_kpca.fit_transform(
        X_scaled
    )

    print(
        f"RBF KPCA Shape : {X_rbf.shape}"
    )

    # ------------------------------------------------------
    # 4. Fit final KMeans
    # ------------------------------------------------------

    print("\nFitting final KMeans...")

    kmeans = KMeans(
        **FINAL_KMEANS_PARAMS
    )

    labels = kmeans.fit_predict(
        X_rbf
    )

    # ------------------------------------------------------
    # 5. Validate cluster distribution
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("CLUSTER DISTRIBUTION")
    print("=" * 70)

    cluster_counts = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
    )

    for cluster, count in (
        cluster_counts.items()
    ):

        percentage = (
            count / len(labels)
        ) * 100

        name = CLUSTER_MAPPING[
            int(cluster)
        ]

        print(
            f"\nCluster {cluster}"
            f" -> {name}"
        )

        print(
            f"Players    : {count}"
        )

        print(
            f"Percentage : {percentage:.2f}%"
        )

    # ------------------------------------------------------
    # 6. Create deployment artifact
    # ------------------------------------------------------

    artifact = {

        # Exact feature order is critical
        "feature_columns":
            list(CLUSTER_FEATURES),

        # Fitted preprocessing
        "scaler":
            scaler,

        # Fitted dimensionality reduction
        "rbf_kpca":
            rbf_kpca,

        # Fitted final clustering model
        "kmeans":
            kmeans,

        # Human-readable mapping
        "cluster_mapping":
            CLUSTER_MAPPING,

        # Metadata
        "n_components":
            PCA_COMPONENTS,

        "model_name":
            "KMeans + RBF Kernel PCA",

        "n_clusters":
            4
    }

    # ------------------------------------------------------
    # 7. Save complete pipeline
    # ------------------------------------------------------

    output_path = (
        MODELS_DIR /
        "final_clustering_pipeline.pkl"
    )

    joblib.dump(
        artifact,
        output_path
    )

    print("\n" + "=" * 70)
    print("FINAL PIPELINE SAVED")
    print("=" * 70)

    print(
        f"\n{output_path}"
    )

    print(
        "\nContains:"
    )

    print(
        "  ✓ Feature column order"
    )

    print(
        "  ✓ Fitted StandardScaler"
    )

    print(
        "  ✓ Fitted RBF Kernel PCA"
    )

    print(
        "  ✓ Final KMeans model"
    )

    print(
        "  ✓ Cluster name mapping"
    )

    return artifact


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    build_final_pipeline()