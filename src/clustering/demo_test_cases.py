"""
demo_test_cases.py
------------------

Generates verified demo test cases for the final football
player clustering pipeline.

Creates:
    10 Experienced Defenders
    10 Elite Forwards
    10 Young Talents
    10 Defensive Specialists

Total:
    40 verified test cases

Each test case is passed through the saved final pipeline:

Raw Features
    -> StandardScaler
    -> RBF Kernel PCA
    -> KMeans
    -> Cluster Name
"""

import joblib
import pandas as pd

from config import (
    DATASET_PATH,
    MODELS_DIR
)


# ==========================================================
# FEATURES
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

    pipeline = joblib.load(model_path)

    return pipeline


# ==========================================================
# PREDICT SINGLE PLAYER
# ==========================================================

def predict_player(player_data, pipeline):

    # Create dataframe in exact feature order
    player_df = pd.DataFrame(
        [{
            feature: player_data[feature]
            for feature in FEATURES
        }]
    )

    # ------------------------------------------------------
    # Scaling
    # ------------------------------------------------------

    X_scaled = pipeline[
        "scaler"
    ].transform(player_df)

    # ------------------------------------------------------
    # RBF Kernel PCA
    # ------------------------------------------------------

    X_rbf = pipeline[
        "rbf_kpca"
    ].transform(X_scaled)

    # ------------------------------------------------------
    # KMeans prediction
    # ------------------------------------------------------

    cluster = int(
        pipeline[
            "kmeans"
        ].predict(X_rbf)[0]
    )

    category = pipeline[
        "cluster_mapping"
    ][cluster]

    return cluster, category


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 80)
    print("GENERATING CLUSTERING DEMO TEST CASES")
    print("=" * 80)


    # ======================================================
    # LOAD PIPELINE
    # ======================================================

    pipeline = load_pipeline()

    cluster_mapping = pipeline[
        "cluster_mapping"
    ]


    # ======================================================
    # LOAD TRAINING DATASET
    # ======================================================

    print("\nLoading training dataset...")

    df = pd.read_csv(
        DATASET_PATH
    )

    print(
        f"Players Loaded : {len(df)}"
    )


    # ======================================================
    # VALIDATE FEATURES
    # ======================================================

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            f"Missing features: {missing_features}"
        )


    # ======================================================
    # RUN TRAINING DATA THROUGH SAVED PIPELINE
    # ======================================================

    print(
        "\nRunning players through "
        "saved clustering pipeline..."
    )

    X = df[
        FEATURES
    ].copy()


    # ------------------------------------------------------
    # Scaling
    # ------------------------------------------------------

    X_scaled = pipeline[
        "scaler"
    ].transform(X)


    # ------------------------------------------------------
    # RBF Kernel PCA
    # ------------------------------------------------------

    X_rbf = pipeline[
        "rbf_kpca"
    ].transform(X_scaled)


    # ------------------------------------------------------
    # KMeans
    # ------------------------------------------------------

    predictions = pipeline[
        "kmeans"
    ].predict(X_rbf)

    df["predicted_cluster"] = predictions


    # ======================================================
    # SHOW FULL CLUSTER DISTRIBUTION
    # ======================================================

    print("\n")
    print("=" * 80)
    print("CLUSTER DISTRIBUTION")
    print("=" * 80)

    cluster_counts = (
        df["predicted_cluster"]
        .value_counts()
        .sort_index()
    )

    for cluster_id, count in cluster_counts.items():

        cluster_id = int(
            cluster_id
        )

        category = cluster_mapping[
            cluster_id
        ]

        percentage = (
            count / len(df)
        ) * 100

        print(
            f"\nCluster {cluster_id}"
            f" -> {category}"
        )

        print(
            f"Players    : {count}"
        )

        print(
            f"Percentage : {percentage:.2f}%"
        )


    # ======================================================
    # GENERATE 10 TEST CASES PER CLUSTER
    # ======================================================

    demo_cases = []

    print("\n")
    print("=" * 80)
    print("GENERATING 10 VERIFIED CASES PER CLUSTER")
    print("=" * 80)


    for cluster_id in [0, 1, 2, 3]:

        category = cluster_mapping[
            cluster_id
        ]

        cluster_players = df[
            df["predicted_cluster"]
            == cluster_id
        ].copy()


        if len(cluster_players) < 10:

            raise ValueError(
                f"Cluster {cluster_id} "
                f"contains fewer than 10 players."
            )


        # --------------------------------------------------
        # Select reproducible random examples
        # --------------------------------------------------

        selected_players = (
            cluster_players
            .sample(
                n=10,
                random_state=42
            )
        )


        print("\n")
        print("-" * 80)

        print(
            f"Cluster {cluster_id}"
            f" -> {category}"
        )

        print("-" * 80)


        # --------------------------------------------------
        # Verify every selected player
        # --------------------------------------------------

        for test_number, (_, row) in enumerate(
            selected_players.iterrows(),
            start=1
        ):

            player_data = {

                feature:
                    float(row[feature])

                for feature in FEATURES
            }


            predicted_cluster, predicted_category = (
                predict_player(
                    player_data,
                    pipeline
                )
            )


            # ------------------------------------------------
            # Verify expected == predicted
            # ------------------------------------------------

            verified = (
                predicted_cluster
                == cluster_id
            )


            print(
                f"Test {test_number:02d}"
                f" | Expected: {cluster_id}"
                f" | Predicted: {predicted_cluster}"
                f" | {predicted_category}"
                f" | {'PASS' if verified else 'FAIL'}"
            )


            # ------------------------------------------------
            # Store result
            # ------------------------------------------------

            demo_case = {

                "test_case":
                    test_number,

                "expected_cluster":
                    cluster_id,

                "expected_category":
                    category,

                "predicted_cluster":
                    predicted_cluster,

                "predicted_category":
                    predicted_category,

                "verified":
                    verified
            }


            # Add all 9 features
            for feature in FEATURES:

                demo_case[
                    feature
                ] = player_data[
                    feature
                ]


            # Add player name if available
            if "player_name" in df.columns:

                demo_case[
                    "player_name"
                ] = row[
                    "player_name"
                ]


            # Add position if available
            if "position_group" in df.columns:

                demo_case[
                    "position_group"
                ] = row[
                    "position_group"
                ]


            demo_cases.append(
                demo_case
            )


    # ======================================================
    # CREATE DEMO DATAFRAME
    # ======================================================

    demo_df = pd.DataFrame(
        demo_cases
    )


    # ======================================================
    # FINAL VERIFICATION
    # ======================================================

    print("\n")
    print("=" * 80)
    print("FINAL DEMO VERIFICATION")
    print("=" * 80)


    total_cases = len(
        demo_df
    )

    passed_cases = int(
        demo_df[
            "verified"
        ].sum()
    )

    failed_cases = (
        total_cases
        - passed_cases
    )


    print(
        f"\nTotal Cases  : {total_cases}"
    )

    print(
        f"Passed       : {passed_cases}"
    )

    print(
        f"Failed       : {failed_cases}"
    )


    # ======================================================
    # TEST CASE COUNT BY CATEGORY
    # ======================================================

    print("\nTest Cases Per Category")
    print("-" * 50)

    print(
        demo_df[
            "expected_category"
        ].value_counts()
    )


    # ======================================================
    # SAVE CSV
    # ======================================================

    output_path = (
        "results/clustering/"
        "final_kmeans/"
        "demo_test_cases.csv"
    )


    demo_df.to_csv(
        output_path,
        index=False
    )


    print("\n")
    print("=" * 80)
    print("DEMO TEST CASES SAVED")
    print("=" * 80)

    print(
        f"\n{output_path}"
    )


    # ======================================================
    # DISPLAY TEST VALUES
    # ======================================================

    print("\n")
    print("=" * 80)
    print("DEMO TEST CASE VALUES")
    print("=" * 80)


    for cluster_id in [0, 1, 2, 3]:

        category = cluster_mapping[
            cluster_id
        ]

        print("\n")
        print(
            f"CLUSTER {cluster_id}"
            f" -> {category}"
        )

        print("-" * 80)

        display_columns = [
            "test_case"
        ] + FEATURES

        cluster_demo = demo_df[
            demo_df["expected_cluster"]
            == cluster_id
        ]

        print(
            cluster_demo[
                display_columns
            ].to_string(
                index=False
            )
        )


    print("\n")
    print("=" * 80)
    print("DEMO GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 80)