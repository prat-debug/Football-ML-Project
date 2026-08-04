"""
preprocessing.py
----------------

Loads cleaned football player dataset and performs preprocessing
required before PCA and clustering.

Author: Team Clustering Module
"""

import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler

from config import (
    DATASET_PATH,
    MODELS_DIR,
    CLUSTER_FEATURES,
    POSITION_COLUMN
)


class DataPreprocessor:
    """
    Handles preprocessing tasks for clustering.

    Steps
    -----
    1. Load cleaned dataset
    2. Validate required columns
    3. Verify no missing values
    4. Standardize numerical features
    5. Save StandardScaler
    """

    def __init__(self):

        self.scaler = StandardScaler()

    # --------------------------------------------------

    def load_dataset(self):

        df = pd.read_csv(DATASET_PATH)

        return df

    # --------------------------------------------------

    def validate_columns(self, df):

        required_columns = CLUSTER_FEATURES + [POSITION_COLUMN]

        missing_columns = [
            col for col in required_columns
            if col not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing columns in dataset: {missing_columns}"
            )

    # --------------------------------------------------

    def preprocess(self):

        df = self.load_dataset()

        self.validate_columns(df)

        X = df[CLUSTER_FEATURES].copy()

        # Since we are using the cleaned dataset,
        # there should not be any missing values.
        if X.isnull().sum().sum() > 0:
            raise ValueError(
                "Cleaned dataset still contains missing values."
            )

        # Standardize features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=CLUSTER_FEATURES
        )

        # Save for cluster interpretation later
        position_group = df[POSITION_COLUMN].copy()

        return X_scaled, position_group, df

    # --------------------------------------------------

    def save_scaler(self):

        output_path = MODELS_DIR / "scaler.pkl"

        joblib.dump(self.scaler, output_path)

        print(f"\nScaler saved to:\n{output_path}")


# ==========================================================
# Utility Function
# ==========================================================

def get_preprocessed_data():
    """
    Returns
    -------
    X_scaled : DataFrame
        Standardized numerical features.

    position_group : Series
        Original player positions.

    original_df : DataFrame
        Complete cleaned dataset.
    """

    processor = DataPreprocessor()

    X_scaled, position_group, original_df = processor.preprocess()

    return X_scaled, position_group, original_df


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    processor = DataPreprocessor()

    X_scaled, positions, original_df = processor.preprocess()

    processor.save_scaler()

    print("\n========== DATA SUMMARY ==========\n")

    print(f"Original Shape : {original_df.shape}")

    print(f"Scaled Shape   : {X_scaled.shape}")

    print("\nFirst 5 Rows\n")

    print(X_scaled.head())

    print("\nPreprocessing completed successfully.")