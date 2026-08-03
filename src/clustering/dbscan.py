"""
dbscan.py
---------

DBSCAN Clustering Module

Performs:

1. Linear PCA + DBSCAN
2. Kernel PCA (RBF) + DBSCAN
3. Kernel PCA (Poly) + DBSCAN
4. Kernel PCA (Sigmoid) + DBSCAN
5. Kernel PCA (Cosine) + DBSCAN

Project Requirement:
Exactly 4 non-noise clusters.

DBSCAN cannot directly specify the number of clusters.
Therefore, eps and min_samples are tuned and only
solutions producing exactly 4 clusters are eligible.

Evaluation:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score
- Noise Points
- Noise Percentage

Saves:
- Metrics CSV
- Cluster plots
- Best DBSCAN model

Author:
Team Clustering Module
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product

from sklearn.cluster import DBSCAN

from linear_pca import get_linear_pca_data
from kernel_pca import get_kernel_pca_data
from evaluation import ClusterEvaluator

from config import (
    MODELS_DIR,
    LINEAR_RESULTS_DIR,
    KERNEL_RESULTS_DIR
)


class DBSCANTrainer:

    def __init__(self):

        self.all_models = {}

        self.best_dataset = None
        self.best_result = None

        # ==================================================
        # TARGET NUMBER OF CLUSTERS
        # ==================================================

        self.target_clusters = 4

        # ==================================================
        # DBSCAN HYPERPARAMETER GRID
        # ==================================================
        #
        # Compact grid to keep runtime reasonable.
        #

        self.param_grid = {

            "eps": [
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                1.0
            ],

            "min_samples": [
                5,
                10,
                15
            ],

            "metric": [
                "euclidean",
                "manhattan"
            ]
        }

    # -----------------------------------------------------

    def parameter_combinations(self):

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

    def count_clusters(self, labels):
        """
        Count DBSCAN clusters excluding noise (-1).
        """

        unique_labels = set(labels)

        unique_labels.discard(-1)

        return len(unique_labels)

    # -----------------------------------------------------

    def train_single_dataset(
        self,
        X,
        dataset_name,
        save_directory
    ):

        print("\n" + "=" * 70)

        print(
            f"Training DBSCAN : {dataset_name}"
        )

        print("=" * 70)

        best_model = None
        best_labels = None
        best_metrics = None
        best_params = None

        best_score = float("-inf")

        all_results = []

        combinations = list(
            self.parameter_combinations()
        )

        total = len(combinations)

        # ==================================================
        # HYPERPARAMETER SEARCH
        # ==================================================

        for index, params in enumerate(
            combinations,
            start=1
        ):

            print(
                f"[{index}/{total}] "
                f"eps={params['eps']}, "
                f"min_samples={params['min_samples']}, "
                f"metric={params['metric']}"
            )

            try:

                model = DBSCAN(

                    eps=params["eps"],

                    min_samples=params["min_samples"],

                    metric=params["metric"],

                    algorithm="auto",

                    n_jobs=-1

                )

                # ------------------------------------------
                # Fit model
                # ------------------------------------------

                labels = model.fit_predict(X)

                # ------------------------------------------
                # Count clusters
                # ------------------------------------------

                number_of_clusters = (
                    self.count_clusters(labels)
                )

                # ------------------------------------------
                # Noise information
                # ------------------------------------------

                noise_points = int(
                    np.sum(labels == -1)
                )

                noise_percentage = (
                    noise_points /
                    len(labels)
                ) * 100

                # ------------------------------------------
                # IMPORTANT OPTIMIZATION
                #
                # Silhouette is expensive.
                #
                # Only calculate full evaluation metrics
                # when DBSCAN actually produces exactly
                # four clusters.
                # ------------------------------------------

                if (
                    number_of_clusters
                    == self.target_clusters
                ):

                    metrics = (
                        ClusterEvaluator.evaluate(
                            X,
                            labels
                        )
                    )

                else:

                    metrics = {

                        "Silhouette Score":
                            np.nan,

                        "Davies-Bouldin Index":
                            np.nan,

                        "Calinski-Harabasz Score":
                            np.nan,

                        "Number of Clusters":
                            number_of_clusters

                    }

                # ------------------------------------------
                # Store every experiment
                # ------------------------------------------

                row = {

                    **params,

                    "algorithm":
                        "auto",

                    **metrics,

                    "Noise Points":
                        noise_points,

                    "Noise Percentage":
                        noise_percentage

                }

                all_results.append(row)

                # ------------------------------------------
                # Reject anything except 4 clusters
                # ------------------------------------------

                if (
                    number_of_clusters
                    != self.target_clusters
                ):

                    continue

                score = metrics[
                    "Silhouette Score"
                ]

                if pd.isna(score):

                    continue

                # ------------------------------------------
                # Select best 4-cluster DBSCAN
                # ------------------------------------------

                if score > best_score:

                    best_score = score

                    best_model = model

                    best_labels = labels

                    best_metrics = {
                        **metrics,
                        "Noise Points":
                            noise_points,
                        "Noise Percentage":
                            noise_percentage
                    }

                    best_params = {
                        **params,
                        "algorithm": "auto"
                    }

            except Exception as error:

                print(
                    f"Skipping invalid combination "
                    f"{params}: {error}"
                )

        # ==================================================
        # SAVE SEARCH RESULTS
        # ==================================================

        results_df = pd.DataFrame(
            all_results
        )

        if not results_df.empty:

            # Four-cluster solutions appear first,
            # followed by highest silhouette score.

            results_df[
                "Target 4 Clusters"
            ] = (

                results_df[
                    "Number of Clusters"
                ]
                == self.target_clusters

            )

            results_df = (
                results_df.sort_values(

                    by=[
                        "Target 4 Clusters",
                        "Silhouette Score"
                    ],

                    ascending=[
                        False,
                        False
                    ],

                    na_position="last"

                )
            )

        os.makedirs(
            save_directory,
            exist_ok=True
        )

        csv_path = os.path.join(

            save_directory,

            "dbscan_metrics.csv"

        )

        results_df.to_csv(

            csv_path,

            index=False

        )

        print(
            f"\nMetrics saved -> {csv_path}"
        )

        # --------------------------------------------------
        # Report whether 4 clusters were found
        # --------------------------------------------------

        if best_model is None:

            print(
                f"\nWARNING: No valid "
                f"{self.target_clusters}-cluster "
                f"DBSCAN solution found for "
                f"{dataset_name}."
            )

        else:

            print(
                f"\nBest 4-cluster solution found:"
            )

            print(
                f"Silhouette Score : "
                f"{best_metrics['Silhouette Score']}"
            )

            print(
                f"Noise Percentage : "
                f"{best_metrics['Noise Percentage']:.2f}%"
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

        if labels is None:
            return

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

            "dbscan_clusters.png"

        )

        plt.savefig(

            plot_path,

            dpi=300,

            bbox_inches="tight"

        )

        plt.close()

        print(
            f"Cluster plot saved -> "
            f"{plot_path}"
        )

    # -----------------------------------------------------

    def train_linear_pca(self):

        # Current get_linear_pca_data()
        # returns:
        #
        # X_pca, positions, df

        X_linear, _, _ = (
            get_linear_pca_data()
        )

        save_dir = os.path.join(

            LINEAR_RESULTS_DIR,

            "dbscan"

        )

        result = (
            self.train_single_dataset(

                X_linear,

                "Linear PCA",

                save_dir

            )
        )

        if result["labels"] is not None:

            self.plot_clusters(

                X_linear,

                result["labels"],

                save_dir,

                "DBSCAN - Linear PCA"

            )

        self.all_models[
            "Linear PCA"
        ] = result

    # -----------------------------------------------------

    def train_kernel_pca(self):

        # get_kernel_pca_data() returns:
        #
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

                "dbscan"

            )

            result = (
                self.train_single_dataset(

                    X_kernel,

                    (
                        "Kernel PCA "
                        f"({kernel_name})"
                    ),

                    save_dir

                )
            )

            if result["labels"] is not None:

                self.plot_clusters(

                    X_kernel,

                    result["labels"],

                    save_dir,

                    (
                        "DBSCAN - "
                        f"{kernel_name.upper()}"
                    )

                )

            self.all_models[
                f"Kernel PCA ({kernel_name})"
            ] = result

    # -----------------------------------------------------

    def compare_models(self):

        print("\n")

        print("=" * 70)

        print(
            "DBSCAN 4-CLUSTER MODEL COMPARISON"
        )

        print("=" * 70)

        best_name = None

        best_score = float("-inf")

        for dataset_name, result in (
            self.all_models.items()
        ):

            metrics = result[
                "metrics"
            ]

            print(
                "\n" + "-" * 60
            )

            print(dataset_name)

            if metrics is None:

                print(
                    "No valid 4-cluster "
                    "solution found."
                )

                continue

            # ------------------------------------------
            # Print parameters
            # ------------------------------------------

            print(
                "\nBest Hyperparameters"
            )

            for key, value in (
                result["params"].items()
            ):

                print(
                    f"{key:<30}: {value}"
                )

            # ------------------------------------------
            # Print metrics
            # ------------------------------------------

            print(
                "\nEvaluation Metrics"
            )

            for key, value in (
                metrics.items()
            ):

                if key == "Noise Percentage":

                    print(
                        f"{key:<30}: "
                        f"{value:.2f}%"
                    )

                else:

                    print(
                        f"{key:<30}: {value}"
                    )

            score = metrics[
                "Silhouette Score"
            ]

            if pd.isna(score):
                continue

            # ------------------------------------------
            # Overall winner
            # ------------------------------------------

            if score > best_score:

                best_score = score

                best_name = dataset_name

        # --------------------------------------------------
        # Check that at least one valid solution exists
        # --------------------------------------------------

        if best_name is None:

            raise RuntimeError(

                "\nDBSCAN could not find an "
                "exactly 4-cluster solution with "
                "the current parameter grid.\n"
                "Do NOT force a result. "
                "The eps/min_samples grid should "
                "be refined instead."

            )

        self.best_dataset = best_name

        self.best_result = (
            self.all_models[
                best_name
            ]
        )

    # -----------------------------------------------------

    def save_best_model(self):

        output_path = os.path.join(

            MODELS_DIR,

            "dbscan_best.pkl"

        )

        joblib.dump(

            self.best_result[
                "model"
            ],

            output_path

        )

        print(
            "\n" + "=" * 70
        )

        print(
            "Best 4-Cluster DBSCAN "
            "Model Saved"
        )

        print("=" * 70)

        print(output_path)

    # -----------------------------------------------------

    def print_summary(self):

        print("\n")

        print("=" * 70)

        print(
            "FINAL DBSCAN RESULT"
        )

        print("=" * 70)

        print(
            f"\nWinner : "
            f"{self.best_dataset}\n"
        )

        print(
            "Best Hyperparameters"
        )

        print("-" * 30)

        for key, value in (
            self.best_result[
                "params"
            ].items()
        ):

            print(
                f"{key:<30}: {value}"
            )

        print(
            "\nEvaluation Metrics"
        )

        print("-" * 30)

        for key, value in (
            self.best_result[
                "metrics"
            ].items()
        ):

            if key == "Noise Percentage":

                print(
                    f"{key:<30}: "
                    f"{value:.2f}%"
                )

            else:

                print(
                    f"{key:<30}: {value}"
                )

        print("=" * 70)

    # -----------------------------------------------------

    def run(self):

        print(
            "\nStarting DBSCAN "
            "4-Cluster Pipeline...\n"
        )

        # Linear PCA
        self.train_linear_pca()

        # Kernel PCA
        self.train_kernel_pca()

        # Compare valid 4-cluster solutions
        self.compare_models()

        # Save winner
        self.save_best_model()

        # Final output
        self.print_summary()


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    trainer = DBSCANTrainer()

    trainer.run()