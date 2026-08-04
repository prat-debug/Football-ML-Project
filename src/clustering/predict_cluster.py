"""
predict_cluster.py
------------------

Predicts the football-player cluster for a new player using
the final deployable clustering pipeline.

Pipeline:
Raw 9 features
    -> StandardScaler
    -> RBF Kernel PCA
    -> KMeans
    -> Cluster Name
"""

import joblib
import pandas as pd

from config import MODELS_DIR


# ==========================================================
# FINAL FEATURE ORDER
# ==========================================================

FEATURES = [
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


# ==========================================================
# LOAD FINAL PIPELINE
# ==========================================================

def load_pipeline():

    model_path = (
        MODELS_DIR /
        "final_clustering_pipeline.pkl"
    )

    print("\nLoading final clustering pipeline...")

    pipeline = joblib.load(
        model_path
    )

    return pipeline


# ==========================================================
# VALIDATE PLAYER DATA
# ==========================================================

def validate_player(player_data):

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in player_data
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    # Make sure all values are numeric
    for feature in FEATURES:

        value = player_data[feature]

        if not isinstance(
            value,
            (int, float)
        ):
            raise ValueError(
                f"{feature} must be numeric. "
                f"Received: {value}"
            )


# ==========================================================
# PREDICT PLAYER CLUSTER
# ==========================================================

def predict_player(
    player_data,
    pipeline
):

    validate_player(
        player_data
    )

    # ------------------------------------------------------
    # 1. Create DataFrame in EXACT training feature order
    # ------------------------------------------------------

    player_df = pd.DataFrame(
        [
            {
                feature:
                    player_data[feature]

                for feature in FEATURES
            }
        ]
    )

    # ------------------------------------------------------
    # 2. Load fitted components
    # ------------------------------------------------------

    scaler = pipeline[
        "scaler"
    ]

    rbf_kpca = pipeline[
        "rbf_kpca"
    ]

    kmeans = pipeline[
        "kmeans"
    ]

    cluster_mapping = pipeline[
        "cluster_mapping"
    ]

    # ------------------------------------------------------
    # 3. Scale using SAVED scaler
    # ------------------------------------------------------

    X_scaled = scaler.transform(
        player_df
    )

    # ------------------------------------------------------
    # 4. RBF Kernel PCA using SAVED transformer
    # ------------------------------------------------------

    X_rbf = rbf_kpca.transform(
        X_scaled
    )

    # ------------------------------------------------------
    # 5. Predict KMeans cluster
    # ------------------------------------------------------

    cluster = int(
        kmeans.predict(
            X_rbf
        )[0]
    )

    # ------------------------------------------------------
    # 6. Convert cluster number -> name
    # ------------------------------------------------------

    category = cluster_mapping[
        cluster
    ]

    return cluster, category


# ==========================================================
# DISPLAY RESULT
# ==========================================================

def display_result(
    player_data,
    cluster,
    category
):

    print("\n")
    print("=" * 60)
    print("PLAYER CLUSTER PREDICTION")
    print("=" * 60)

    print("\nPlayer Features")
    print("-" * 40)

    for feature in FEATURES:

        print(
            f"{feature:<25}: "
            f"{player_data[feature]}"
        )

    print("\nPrediction")
    print("-" * 40)

    print(
        f"Predicted Cluster : {cluster}"
    )

    print(
        f"Player Category   : {category}"
    )

    print("=" * 60)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    pipeline = load_pipeline()

    # ======================================================
    # TEST PLAYER
    # ======================================================
    #
    # Replace these values with another player's values
    # whenever you want to test prediction.
    #
    # IMPORTANT:
    # Use the same units/definitions as the training dataset.
    # ======================================================

    new_player = {

        "age": 25,

        "height_in_cm": 180,

        "matches_played": 30,

        "minutes_per_match": 75,

        "goals_per_90": 0.60,

        "assists_per_90": 0.20,

        "estimated_xg": 15.0,

        "estimated_xa": 6.0,

        "physicality_score": 70.0

    }

    cluster, category = predict_player(
        new_player,
        pipeline
    )

    display_result(
        new_player,
        cluster,
        category
    )