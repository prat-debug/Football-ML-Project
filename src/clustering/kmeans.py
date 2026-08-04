"""
kmeans.py
---------

KMeans Clustering Module

Performs:

1. Linear PCA + KMeans
2. Kernel PCA (RBF) + KMeans
3. Kernel PCA (Poly) + KMeans
4. Kernel PCA (Sigmoid) + KMeans
5. Kernel PCA (Cosine) + KMeans

Project Requirement:
Exactly 4 clusters.

Performs hyperparameter tuning.

Evaluation:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score

Saves:
- Metrics CSV
- Cluster plots
- Best KMeans model

Author:
Team Clustering Module
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product

from sklearn.cluster import KMeans

from linear_pca import get_linear_pca_data
from kernel_pca import get_kernel_pca_data
from evaluation import ClusterEvaluator

from config import (
    RANDOM_STATE,
    MODELS_DIR,
    LINEAR_RESULTS_DIR,
    KERNEL_RESULTS_DIR
)


class KMeansTrainer:
    """
    Handles:

    - KMeans hyperparameter tuning
    - Linear PCA experiments
    - Kernel PCA experiments
    - Model comparison
    - Best model saving
    """

    def __init__(self):

        self.all_models = {}

        self.best_dataset = None
        self.best_result = None

        # ==================================================
        # KMEANS HYPERPARAMETER GRID
        # ==================================================

        self.param_grid = {

            # Project requirement:
            # exactly 4 player clusters
            "n_clusters": [4],

            "init": [
                "k-means++",
                "random"
            ],

            "n_init": [
                10,
                20
            ],

            "max_iter": [
                300,
                500
            ],

            "algorithm": [
                "lloyd",
                "elkan"
            ]
        }

    # -----------------------------------------------------

    def parameter_combinations(self):
        """
        Generate all KMeans hyperparameter combinations.
        """

        keys = list(
            self.param_grid.keys()
        )

        values = list(
            self.param_grid.values()
        )

        for combination in product(*values):

            yield dict(
                zip(keys, combination)
            )

    # -----------------------------------------------------

    def train_single_dataset(
        self,
        X,
        dataset_name,
        save_directory
    ):
        """
        Tune KMeans on one PCA representation.
        """

        print("\n" + "=" * 70)

        print(
            f"Training KMeans : {dataset_name}"
        )

        print("=" * 70)

        best_model = None
        best_labels = None
        best_metrics = None
        best_params = None

        best_score = float("-inf")

        all_results = []

        # ==================================================
        # HYPERPARAMETER TUNING
        # ==================================================

        for params in self.parameter_combinations():

            try:

                model = KMeans(

                    random_state=RANDOM_STATE,

                    **params

                )

                # ------------------------------------------
                # Train model
                # ------------------------------------------

                labels = model.fit_predict(X)

                # ------------------------------------------
                # Evaluate clustering
                # ------------------------------------------

                metrics = ClusterEvaluator.evaluate(
                    X,
                    labels
                )

                # ------------------------------------------
                # Store result
                # ------------------------------------------

                row = {

                    **params,

                    **metrics,

                    "Inertia":
                        model.inertia_,

                    "Iterations":
                        model.n_iter_

                }

                all_results.append(row)

                # ------------------------------------------
                # Ensure exactly 4 clusters
                # ------------------------------------------

                if (
                    metrics["Number of Clusters"]
                    != 4
                ):

                    continue

                # ------------------------------------------
                # Select best configuration
                # ------------------------------------------

                score = metrics[
                    "Silhouette Score"
                ]

                if pd.isna(score):
                    continue

                if score > best_score:

                    best_score = score

                    best_model = model

                    best_labels = labels

                    best_metrics = metrics

                    best_params = params.copy()

            except Exception as error:

                print(
                    f"Skipping invalid combination "
                    f"{params}: {error}"
                )

        # ==================================================
        # SAVE TUNING RESULTS
        # ==================================================

        results_df = pd.DataFrame(
            all_results
        )

        if not results_df.empty:

            results_df = results_df.sort_values(

                by="Silhouette Score",

                ascending=False,

                na_position="last"

            )

        os.makedirs(
            save_directory,
            exist_ok=True
        )

        csv_path = os.path.join(

            save_directory,

            "kmeans_metrics.csv"

        )

        results_df.to_csv(

            csv_path,

            index=False

        )

        print(
            f"\nMetrics saved -> {csv_path}"
        )

        return {

            "model":
                best_model,

            "labels":
                best_labels,

            "metrics":
                best_metrics,

            "params":
                best_params,

            "results":
                results_df

        }

    # -----------------------------------------------------

    def plot_clusters(
        self,
        X,
        labels,
        save_directory,
        title
    ):
        """
        Plot clusters using first two PCA components.
        """

        os.makedirs(
            save_directory,
            exist_ok=True
        )

        if hasattr(X, "values"):

            X_plot = X.values

        else:

            X_plot = X

        plt.figure(
            figsize=(8, 6)
        )

        scatter = plt.scatter(

            X_plot[:, 0],

            X_plot[:, 1],

            c=labels,

            cmap="tab10",

            s=20,

            alpha=0.8

        )

        plt.title(title)

        plt.xlabel(
            "Principal Component 1"
        )

        plt.ylabel(
            "Principal Component 2"
        )

        plt.colorbar(
            scatter,
            label="Cluster"
        )

        plot_path = os.path.join(

            save_directory,

            "kmeans_clusters.png"

        )

        plt.savefig(

            plot_path,

            dpi=300,

            bbox_inches="tight"

        )

        plt.close()

        print(
            f"Cluster plot saved -> {plot_path}"
        )

    # -----------------------------------------------------

    def train_linear_pca(self):
        """
        Train KMeans using Linear PCA.
        """

        # Your current function returns:
        # X_pca, positions, df

        X_linear, _, _ = (
            get_linear_pca_data()
        )

        save_dir = os.path.join(

            LINEAR_RESULTS_DIR,

            "kmeans"

        )

        linear_result = (
            self.train_single_dataset(

                X_linear,

                "Linear PCA",

                save_dir

            )
        )

        if linear_result["labels"] is not None:

            self.plot_clusters(

                X_linear,

                linear_result["labels"],

                save_dir,

                "KMeans - Linear PCA"

            )

        self.all_models[
            "Linear PCA"
        ] = linear_result

    # -----------------------------------------------------

    def train_kernel_pca(self):
        """
        Train KMeans using all Kernel PCA representations.
        """

        # get_kernel_pca_data() returns:
        # transformed, positions, df

        kernel_datasets, _, _ = (
            get_kernel_pca_data()
        )

        for kernel_name, X_kernel in (
            kernel_datasets.items()
        ):

            save_dir = os.path.join(

                KERNEL_RESULTS_DIR,

                kernel_name,

                "kmeans"

            )

            kernel_result = (
                self.train_single_dataset(

                    X_kernel,

                    f"Kernel PCA ({kernel_name})",

                    save_dir

                )
            )

            if kernel_result["labels"] is not None:

                self.plot_clusters(

                    X_kernel,

                    kernel_result["labels"],

                    save_dir,

                    (
                        "KMeans - "
                        f"{kernel_name.upper()}"
                    )

                )

            self.all_models[
                f"Kernel PCA ({kernel_name})"
            ] = kernel_result

    # -----------------------------------------------------

    def compare_models(self):
        """
        Compare best KMeans result from every
        PCA representation.
        """

        best_name = None

        best_score = float("-inf")

        print("\n")

        print("=" * 70)

        print(
            "KMEANS MODEL COMPARISON"
        )

        print("=" * 70)

        for dataset_name, result in (
            self.all_models.items()
        ):

            metrics = result[
                "metrics"
            ]

            if metrics is None:
                continue

            score = metrics[
                "Silhouette Score"
            ]

            if pd.isna(score):
                continue

            print(
                "\n" + "-" * 60
            )

            print(dataset_name)

            # ------------------------------------------
            # Evaluation metrics
            # ------------------------------------------

            for key, value in (
                metrics.items()
            ):

                print(
                    f"{key:<30}: {value}"
                )

            # ------------------------------------------
            # Best parameters
            # ------------------------------------------

            print(
                "\nBest Parameters"
            )

            for key, value in (
                result["params"].items()
            ):

                print(
                    f"{key:<30}: {value}"
                )

            print(
                f"{'Inertia':<30}: "
                f"{result['model'].inertia_}"
            )

            print(
                f"{'Iterations':<30}: "
                f"{result['model'].n_iter_}"
            )

            # ------------------------------------------
            # Select winner
            # ------------------------------------------

            if score > best_score:

                best_score = score

                best_name = dataset_name

        # --------------------------------------------------
        # Ensure valid model exists
        # --------------------------------------------------

        if best_name is None:

            raise RuntimeError(
                "KMeans could not produce "
                "a valid 4-cluster solution."
            )

        self.best_dataset = best_name

        self.best_result = (
            self.all_models[
                best_name
            ]
        )

    # -----------------------------------------------------

    def save_best_model(self):
        """
        Save overall best KMeans model.
        """

        output_path = os.path.join(

            MODELS_DIR,

            "kmeans_best.pkl"

        )

        joblib.dump(

            self.best_result["model"],

            output_path

        )

        print(
            "\n" + "=" * 70
        )

        print(
            "Best KMeans Model Saved"
        )

        print("=" * 70)

        print(output_path)

    # -----------------------------------------------------

    def print_summary(self):
        """
        Print final KMeans result.
        """

        print("\n")

        print("=" * 70)

        print(
            "FINAL KMEANS RESULT"
        )

        print("=" * 70)

        print(
            f"\nWinner : "
            f"{self.best_dataset}\n"
        )

        # ==================================================
        # HYPERPARAMETERS
        # ==================================================

        print(
            "Best Hyperparameters"
        )

        print("-" * 30)

        best_params = (
            self.best_result[
                "params"
            ]
        )

        for key, value in (
            best_params.items()
        ):

            print(
                f"{key:<30}: {value}"
            )

        # ==================================================
        # METRICS
        # ==================================================

        print(
            "\nEvaluation Metrics"
        )

        print("-" * 30)

        metrics = (
            self.best_result[
                "metrics"
            ]
        )

        for key, value in (
            metrics.items()
        ):

            print(
                f"{key:<30}: {value}"
            )

        # ==================================================
        # KMEANS INFORMATION
        # ==================================================

        model = (
            self.best_result[
                "model"
            ]
        )

        print(
            "\nKMeans Information"
        )

        print("-" * 30)

        print(
            f"{'Inertia':<30}: "
            f"{model.inertia_}"
        )

        print(
            f"{'Iterations':<30}: "
            f"{model.n_iter_}"
        )

        print("=" * 70)

    # -----------------------------------------------------

    def run(self):
        """
        Run complete KMeans pipeline.
        """

        print(
            "\nStarting KMeans "
            "4-Cluster Pipeline...\n"
        )

        # Linear PCA
        self.train_linear_pca()

        # Kernel PCA
        self.train_kernel_pca()

        # Compare representations
        self.compare_models()

        # Save winner
        self.save_best_model()

        # Final summary
        self.print_summary()


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    trainer = KMeansTrainer()

    trainer.run()