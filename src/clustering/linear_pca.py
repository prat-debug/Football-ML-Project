"""
linear_pca.py
-------------

Performs Linear PCA on standardized football player features.

Steps
-----
1. Load scaled data
2. Fit PCA
3. Determine optimal number of components (95% variance)
4. Transform data
5. Save PCA model
6. Save explained variance metrics
7. Generate plots

Author : Team Clustering Module
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA

from preprocessing import get_preprocessed_data
from config import (
    MODELS_DIR,
    LINEAR_RESULTS_DIR,
    RANDOM_STATE
)


class LinearPCATransformer:

    def __init__(self, variance_threshold=0.95):

        self.variance_threshold = variance_threshold

        self.pca = None

        self.best_components = None

    # --------------------------------------------------------

    def fit(self, X):

        """
        Fit PCA using all components.
        """

        full_pca = PCA(random_state=RANDOM_STATE)

        full_pca.fit(X)

        cumulative_variance = full_pca.explained_variance_ratio_.cumsum()

        self.best_components = (
            cumulative_variance >= self.variance_threshold
        ).argmax() + 1

        print(f"\nOptimal Components : {self.best_components}")

        self.pca = PCA(
            n_components=self.best_components,
            random_state=RANDOM_STATE
        )

        X_pca = self.pca.fit_transform(X)

        return X_pca, full_pca

    # --------------------------------------------------------

    def save_model(self):

        output_path = MODELS_DIR / "linear_pca.pkl"

        joblib.dump(self.pca, output_path)

        print(f"\nLinear PCA saved to:\n{output_path}")

    # --------------------------------------------------------

    def save_variance_csv(self, full_pca):

        df = pd.DataFrame({

            "Component":
                range(
                    1,
                    len(full_pca.explained_variance_ratio_) + 1
                ),

            "Explained Variance":
                full_pca.explained_variance_ratio_,

            "Cumulative Variance":
                full_pca.explained_variance_ratio_.cumsum()

        })

        output_path = (
            LINEAR_RESULTS_DIR /
            "explained_variance.csv"
        )

        df.to_csv(output_path, index=False)

        print(f"\nVariance CSV saved to:\n{output_path}")

    # --------------------------------------------------------

    def variance_plot(self, full_pca):

        cumulative = (
            full_pca.explained_variance_ratio_.cumsum()
        )

        plt.figure(figsize=(8,5))

        plt.plot(
            range(1, len(cumulative)+1),
            cumulative,
            marker="o"
        )

        plt.axhline(
            y=self.variance_threshold,
            color="red",
            linestyle="--"
        )

        plt.xlabel("Number of Components")

        plt.ylabel("Cumulative Explained Variance")

        plt.title("Linear PCA Explained Variance")

        plt.grid(True)

        output_path = (
            LINEAR_RESULTS_DIR /
            "explained_variance.png"
        )

        plt.savefig(output_path, dpi=300)

        plt.close()

        print(f"\nVariance plot saved to:\n{output_path}")

    # --------------------------------------------------------

    def pca_scatter(self, X_pca):

        if X_pca.shape[1] < 2:

            return

        plt.figure(figsize=(8,6))

        plt.scatter(
            X_pca[:,0],
            X_pca[:,1],
            s=10,
            alpha=0.5
        )

        plt.xlabel("PC1")

        plt.ylabel("PC2")

        plt.title("Linear PCA Projection")

        output_path = (
            LINEAR_RESULTS_DIR /
            "linear_pca_scatter.png"
        )

        plt.savefig(output_path, dpi=300)

        plt.close()

        print(f"\nScatter plot saved to:\n{output_path}")


# ===========================================================

def get_linear_pca_data():

    X_scaled, positions, df = get_preprocessed_data()

    transformer = LinearPCATransformer()

    X_pca, full_pca = transformer.fit(X_scaled)

    return X_pca, positions, df


# ===========================================================

if __name__ == "__main__":

    X_scaled, positions, df = get_preprocessed_data()

    transformer = LinearPCATransformer()

    X_pca, full_pca = transformer.fit(X_scaled)

    transformer.save_model()

    transformer.save_variance_csv(full_pca)

    transformer.variance_plot(full_pca)

    transformer.pca_scatter(X_pca)

    print("\nLinear PCA completed successfully.")

    print(f"\nReduced Shape : {X_pca.shape}")