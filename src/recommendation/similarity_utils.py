import joblib
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import NearestNeighbors

from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "cleaned"
    / "clean_player_features.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ==========================================================
# FEATURES
# ==========================================================

NUMERICAL_FEATURES = [

    "age",

    "height_in_cm",

    "goals_per_90",

    "assists_per_90",

    "estimated_xg",

    "estimated_xa",

    "minutes_per_match",

    "physicality_score",

    "performance_score",

    "talent_score"

]

CATEGORICAL_FEATURES = [

    "sub_position"

]

PLAYER_COLUMN = "player_name"

# ==========================================================
# LOAD DATA
# ==========================================================

def load_dataset():

    df = pd.read_csv(DATA_FILE, encoding="utf-8")

    return df


# ==========================================================
# PREPARE FEATURES
# ==========================================================

def prepare_features(df):

    columns = (
        [PLAYER_COLUMN]
        + NUMERICAL_FEATURES
        + CATEGORICAL_FEATURES
    )

    df = df[columns].copy()

    player_names = df[PLAYER_COLUMN].tolist()

    X = df.drop(columns=[PLAYER_COLUMN])

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES
            ),

            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                CATEGORICAL_FEATURES
            )

        ]

    )

    X_processed = preprocessor.fit_transform(X)

    # Convert sparse matrix to dense matrix if required
    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    return (
        player_names,
        X_processed,
        preprocessor
    )

# ==========================================================
# KNN
# ==========================================================

def train_knn(features):

    best_model = None

    best_params = None

    best_average_distance = float("inf")

    for metric in ["euclidean", "manhattan"]:

        for neighbors in [3,5,7,9,11]:

            model = NearestNeighbors(

                n_neighbors=neighbors,

                metric=metric

            )

            model.fit(features)

            distances, _ = model.kneighbors(features)

            score = np.mean(distances)

            if score < best_average_distance:

                best_average_distance = score

                best_model = model

                best_params = {

                    "metric": metric,

                    "n_neighbors": neighbors

                }

    return best_model, best_params, best_average_distance

# ==========================================================
# RECOMMENDATION ENGINE
# ==========================================================

class RecommendationEngine:

    def __init__(
    self,
    df,
    player_names,
    preprocessor,
    features,
    model,
    best_params
    ):
        self.df = df

        self.player_names = list(player_names)

        self.preprocessor = preprocessor

        self.features = features

        self.model = model

    def recommend(

        self,

        player_name,

        method="knn",

        top_n=10

    ):

        if player_name not in self.player_names:

            raise ValueError(

                f"{player_name} not found."

            )

        idx = self.player_names.index(

            player_name

        )

        method = method.lower()

        if method == "knn":
            distances, indices = self.model.kneighbors(

                self.features[idx].reshape(1,-1),

                n_neighbors=top_n+1

            )

            results = []
            tableau_results = []

            for distance, index in zip(

                distances[0][1:],

                indices[0][1:]

            ):
                results.append({

            "Rank": len(results) + 1,

            "Player": self.player_names[index],

            "Distance": round(distance, 4)

            })
                player_info = self.df.iloc[index]

                tableau_results.append({

                    "Query Player": player_name,

                    "Recommended Player": self.player_names[index],

                    "Rank": len(results),

                    "Method": "KNN",

                    "Distance": round(distance, 4),

                    "Age": player_info["age"],

                    "Height (cm)": player_info["height_in_cm"],

                    "Position": player_info["sub_position"],

                    "Goals/90": player_info["goals_per_90"],

                    "Assists/90": player_info["assists_per_90"],

                    "Estimated xG": player_info["estimated_xg"],

                    "Estimated xA": player_info["estimated_xa"],

                    "Minutes/Match": player_info["minutes_per_match"],

                    "Physicality Score": player_info["physicality_score"],

                    "Performance Score": player_info["performance_score"],

                    "Talent Score": player_info["talent_score"]

                })
            tableau_df = pd.DataFrame(tableau_results)
            print(tableau_df.head())
            output_dir = Path("results/tableau")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / "recommendation_dashboard_data.csv"

            tableau_df.to_csv(
                output_file,
                mode="a",
                header=not output_file.exists(),
                index=False
            )

            print(f"Tableau data exported to: {output_file}")
            return pd.DataFrame(results)


        elif method == "cosine":
            query_vector = self.features[idx].reshape(1, -1)

            similarities = cosine_similarity(
                query_vector,
                self.features
            ).flatten()

            order = np.argsort(similarities)[::-1]

            results = []
            tableau_results = []

            for index in order[1:top_n+1]:

                results.append({

                    "Rank": len(results) + 1,

                    "Player": self.player_names[index],

                    "Similarity": round(similarities[index], 4)

                })
                player_info = self.df.iloc[index]

                tableau_results.append({

                    "Query Player": player_name,

                    "Recommended Player": self.player_names[index],

                    "Rank": len(results),

                    "Method": "Cosine Similarity",

                    "Similarity": round(similarities[index], 4),

                    "Age": player_info["age"],

                    "Height (cm)": player_info["height_in_cm"],

                    "Position": player_info["sub_position"],

                    "Goals/90": player_info["goals_per_90"],

                    "Assists/90": player_info["assists_per_90"],

                    "Estimated xG": player_info["estimated_xg"],

                    "Estimated xA": player_info["estimated_xa"],

                    "Minutes/Match": player_info["minutes_per_match"],

                    "Physicality Score": player_info["physicality_score"],

                    "Performance Score": player_info["performance_score"],

                    "Talent Score": player_info["talent_score"]

                })
            tableau_df = pd.DataFrame(tableau_results)
            print(tableau_df.head())
            output_dir = Path("results/tableau")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / "recommendation_dashboard_data.csv"

            tableau_df.to_csv(
                output_file,
                mode="a",
                header=not output_file.exists(),
                index=False
            )

            print(f"Tableau data exported to: {output_file}")
            return pd.DataFrame(results)


        else:
            raise ValueError(
        "Method must be one of: knn, cosine."
        )

# ==========================================================
# SAVE MODEL
# ==========================================================

def save_engine(

    engine,

    filename="player_similarity.pkl"

):

    output_file = (

        MODEL_DIR

        / filename

    )

    joblib.dump(

        engine,

        output_file

    )

    print(

        f"\nModel saved to:\n{output_file}"

    )


# ==========================================================
# LOAD MODEL
# ==========================================================

def load_engine(

    filename="player_similarity.pkl"

):

    input_file = (

        MODEL_DIR

        / filename

    )

    return joblib.load(

        input_file

    )