# ==========================================================
# IMPORTS
# ==========================================================

import pandas as pd
import numpy as np

from similarity_utils import *

# ==========================================================
# RESULTS DIRECTORY
# ==========================================================

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ==========================================================
# HEADER
# ==========================================================

print("=" * 80)
print("SIMILAR PLAYER RECOMMENDATION TRAINING")
print("=" * 80)

# ==========================================================
# LOAD DATASET
# ==========================================================

print("\nLoading dataset...")

df = load_dataset()

print(f"Dataset Shape : {df.shape}")

# ==========================================================
# FEATURE PREPARATION
# ==========================================================

print("\nPreparing features...")

player_names, features, preprocessor = prepare_features(df)

print("Feature preparation completed.")

# ==========================================================
# TRAIN KNN
# ==========================================================

print("\nSearching Best KNN Parameters...")

knn_model, best_params, average_distance = train_knn(features)

print("\nBest Parameters")
print("-" * 40)

print(f"Metric          : {best_params['metric']}")
print(f"Neighbors       : {best_params['n_neighbors']}")
print(f"Average Distance: {average_distance:.6f}")

# ==========================================================
# CREATE PERFORMANCE SUMMARY
# ==========================================================

performance_summary = pd.DataFrame({

    "Metric": [

        "Recommendation Algorithm",
        "Distance Metric",
        "Number of Neighbors",
        "Average Neighbor Distance"

    ],

    "Value": [

        "KNN",
        best_params["metric"],
        best_params["n_neighbors"],
        round(average_distance, 6)

    ]

})

performance_file = (
    RESULTS_DIR /
    "recommendation_model_performance.csv"
)

performance_summary.to_csv(
    performance_file,
    index=False
)

print("\nPerformance summary saved.")

print(performance_file)

# ==========================================================
# CREATE RECOMMENDATION ENGINE
# ==========================================================

print("\nCreating Recommendation Engine...")

engine = RecommendationEngine(
    df=df,

    player_names=player_names,

    preprocessor=preprocessor,

    features=features,

    model=knn_model,

    best_params=best_params

)

# ==========================================================
# SAVE MODEL
# ==========================================================

save_engine(

    engine,

    "player_similarity.pkl"

)

print("\nModel saved to:")

print(

    MODEL_DIR /

    "player_similarity.pkl"

)

# ==========================================================
# FINAL SUMMARY
# ==========================================================

print("\n" + "=" * 80)

print("TRAINING COMPLETED SUCCESSFULLY")

print("=" * 80)

print("\nOutput Files")

print("-" * 40)

print("1.")

print(performance_file)

print("\n2.")

print(

    MODEL_DIR /

    "player_similarity.pkl"

)

print("\nBest Model")

print("-" * 40)

print("Recommendation Algorithm : KNN")

print(f"Distance Metric          : {best_params['metric']}")

print(f"Neighbors                : {best_params['n_neighbors']}")

print(f"Average Distance         : {average_distance:.6f}")

print("\nDone.")