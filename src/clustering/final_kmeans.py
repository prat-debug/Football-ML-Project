"""
final_kmeans.py
---------------

Final targeted KMeans tuning.

Representation:
    RBF Kernel PCA

Requirement:
    Exactly 4 meaningful clusters.

Selection considers:
    - Silhouette Score
    - Davies-Bouldin Index
    - Calinski-Harabasz Score
    - Cluster size validation

Author:
Team Clustering Module
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product
from sklearn.cluster import KMeans

from kernel_pca import get_kernel_pca_data
from evaluation import ClusterEvaluator

from config import (
    RANDOM_STATE,
    MODELS_DIR,
    RESULTS_DIR
)


class FinalKMeansTrainer:

    def __init__(self):

        # ==================================================
        # PROJECT REQUIREMENTS
        # ==================================================

        self.target_clusters = 4

        # Practical segmentation validation thresholds
        self.min_cluster_percentage = 1.0
        self.max_cluster_percentage = 85.0

        # ==================================================
        # TARGETED KMEANS GRID
        # ==================================================

        self.param_grid = {

            "init": [
                "k-means++",
                "random"
            ],

            "n_init": [
                10,
                20,
                30
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

        self.best_model = None
        self.best_labels = None
        self.best_metrics = None
        self.best_params = None
        self.best_score = float("-inf")

        self.results = []

    # ------------------------------------------------------

    def load_rbf_data(self):

        print("\n" + "=" * 70)
        print("LOADING RBF KERNEL PCA DATA")
        print("=" * 70)

        transformed, positions, df = (
            get_kernel_pca_data()
        )

        if "rbf" not in transformed:

            raise KeyError(
                "RBF Kernel PCA representation not found."
            )

        self.X = transformed["rbf"]

        self.positions = positions
        self.df = df

        print(
            f"\nRBF Kernel PCA Shape : "
            f"{self.X.shape}"
        )

    # ------------------------------------------------------

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

    # ------------------------------------------------------

    def get_cluster_distribution(
        self,
        labels
    ):

        labels = np.asarray(labels)

        total = len(labels)

        distribution = {}

        for cluster in np.unique(labels):

            count = int(
                np.sum(labels == cluster)
            )

            percentage = (
                count / total
            ) * 100

            distribution[int(cluster)] = {

                "count":
                    count,

                "percentage":
                    percentage
            }

        return distribution

    # ------------------------------------------------------

    def validate_distribution(
        self,
        distribution
    ):

        # Must have exactly 4 clusters

        if (
            len(distribution)
            != self.target_clusters
        ):

            return (
                False,
                "Not exactly 4 clusters"
            )

        percentages = [

            info["percentage"]

            for info in distribution.values()

        ]

        smallest = min(
            percentages
        )

        largest = max(
            percentages
        )

        # Tiny cluster

        if (
            smallest
            < self.min_cluster_percentage
        ):

            return (
                False,
                (
                    "Tiny cluster detected "
                    f"({smallest:.2f}%)"
                )
            )

        # Dominant cluster

        if (
            largest
            > self.max_cluster_percentage
        ):

            return (
                False,
                (
                    "Dominant cluster detected "
                    f"({largest:.2f}%)"
                )
            )

        return (
            True,
            "Valid"
        )

    # ------------------------------------------------------

    def train(self):

        print("\n" + "=" * 70)
        print("FINAL KMEANS + RBF TUNING")
        print("=" * 70)

        combinations = list(
            self.parameter_combinations()
        )

        total = len(combinations)

        print(
            f"\nTotal combinations : {total}"
        )

        for index, params in enumerate(
            combinations,
            start=1
        ):

            print(
                f"\n[{index}/{total}] "
                f"{params}"
            )

            model = KMeans(

                n_clusters=
                    self.target_clusters,

                random_state=
                    RANDOM_STATE,

                **params

            )

            labels = model.fit_predict(
                self.X
            )

            # ------------------------------------------
            # Cluster distribution
            # ------------------------------------------

            distribution = (
                self.get_cluster_distribution(
                    labels
                )
            )

            valid, reason = (
                self.validate_distribution(
                    distribution
                )
            )

            percentages = [

                info["percentage"]

                for info
                in distribution.values()

            ]

            smallest = min(
                percentages
            )

            largest = max(
                percentages
            )

            # ------------------------------------------
            # Evaluation metrics
            # ------------------------------------------

            metrics = (
                ClusterEvaluator.evaluate(
                    self.X,
                    labels
                )
            )

            # ------------------------------------------
            # Store experiment
            # ------------------------------------------

            row = {

                "n_clusters":
                    self.target_clusters,

                **params,

                **metrics,

                "Inertia":
                    model.inertia_,

                "Iterations":
                    model.n_iter_,

                "Smallest Cluster %":
                    smallest,

                "Largest Cluster %":
                    largest,

                "Valid Segmentation":
                    valid,

                "Validation Reason":
                    reason

            }

            # Add individual cluster sizes

            for cluster, info in (
                distribution.items()
            ):

                row[
                    f"Cluster {cluster} Size"
                ] = info["count"]

                row[
                    f"Cluster {cluster} %"
                ] = info["percentage"]

            self.results.append(
                row
            )

            print(
                f"Silhouette : "
                f"{metrics['Silhouette Score']:.6f}"
            )

            print(
                f"Smallest Cluster : "
                f"{smallest:.2f}%"
            )

            print(
                f"Largest Cluster  : "
                f"{largest:.2f}%"
            )

            print(
                f"Valid : {valid}"
            )

            # ------------------------------------------
            # Only valid segmentation can win
            # ------------------------------------------

            if not valid:

                continue

            score = metrics[
                "Silhouette Score"
            ]

            if pd.isna(score):

                continue

            # ------------------------------------------
            # Select best valid configuration
            # ------------------------------------------

            if score > self.best_score:

                self.best_score = score

                self.best_model = model

                self.best_labels = labels

                self.best_metrics = metrics

                self.best_params = (
                    params.copy()
                )

                self.best_distribution = (
                    distribution.copy()
                )

    # ------------------------------------------------------

    def save_tuning_results(self):

        output_dir = os.path.join(

            RESULTS_DIR,

            "final_kmeans"

        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        results_df = pd.DataFrame(
            self.results
        )

        results_df = (
            results_df.sort_values(

                by=[
                    "Valid Segmentation",
                    "Silhouette Score"
                ],

                ascending=[
                    False,
                    False
                ]

            )
        )

        output_path = os.path.join(

            output_dir,

            "final_kmeans_rbf_tuning.csv"

        )

        results_df.to_csv(

            output_path,

            index=False

        )

        print(
            f"\nTuning results saved -> "
            f"{output_path}"
        )

    # ------------------------------------------------------

    def save_cluster_assignments(self):

        """
        Save row-level cluster labels.

        This is useful later for cluster profiling.
        """

        output_dir = os.path.join(

            RESULTS_DIR,

            "final_kmeans"

        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        output_df = self.df.copy()

        output_df[
            "cluster"
        ] = self.best_labels

        output_path = os.path.join(

            output_dir,

            "player_cluster_assignments.csv"

        )

        output_df.to_csv(

            output_path,

            index=False

        )

        print(
            f"Player cluster assignments saved -> "
            f"{output_path}"
        )

    # ------------------------------------------------------

    def save_model(self):

        if self.best_model is None:

            raise RuntimeError(
                "No valid KMeans model was found."
            )

        output_path = os.path.join(

            MODELS_DIR,

            "final_kmeans_rbf.pkl"

        )

        joblib.dump(

            self.best_model,

            output_path

        )

        print(
            f"\nFinal KMeans model saved -> "
            f"{output_path}"
        )

    # ------------------------------------------------------

    def plot_clusters(self):

        output_dir = os.path.join(

            RESULTS_DIR,

            "final_kmeans"

        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        plt.figure(
            figsize=(9, 7)
        )

        scatter = plt.scatter(

            self.X[:, 0],

            self.X[:, 1],

            c=self.best_labels,

            cmap="tab10",

            s=18,

            alpha=0.7

        )

        plt.xlabel(
            "RBF KPCA Component 1"
        )

        plt.ylabel(
            "RBF KPCA Component 2"
        )

        plt.title(
            "Final KMeans Clustering - RBF Kernel PCA"
        )

        plt.colorbar(
            scatter,
            label="Cluster"
        )

        output_path = os.path.join(

            output_dir,

            "final_kmeans_rbf_clusters.png"

        )

        plt.savefig(

            output_path,

            dpi=300,

            bbox_inches="tight"

        )

        plt.close()

        print(
            f"Cluster plot saved -> "
            f"{output_path}"
        )

    # ------------------------------------------------------

    def print_summary(self):

        print("\n")
        print("=" * 70)
        print("FINAL KMEANS + RBF RESULT")
        print("=" * 70)

        print(
            "\nRepresentation : "
            "Kernel PCA (RBF)"
        )

        print(
            "\nBest Hyperparameters"
        )

        print("-" * 30)

        print(
            f"{'n_clusters':<30}: "
            f"{self.target_clusters}"
        )

        for key, value in (
            self.best_params.items()
        ):

            print(
                f"{key:<30}: {value}"
            )

        print(
            "\nEvaluation Metrics"
        )

        print("-" * 30)

        for key, value in (
            self.best_metrics.items()
        ):

            print(
                f"{key:<30}: {value}"
            )

        print(
            "\nCluster Distribution"
        )

        print("-" * 30)

        for cluster, info in sorted(
            self.best_distribution.items()
        ):

            print(
                f"Cluster {cluster:<3} "
                f": {info['count']:>6} players "
                f"({info['percentage']:.2f}%)"
            )

        print(
            f"\n{'Inertia':<30}: "
            f"{self.best_model.inertia_}"
        )

        print(
            f"{'Iterations':<30}: "
            f"{self.best_model.n_iter_}"
        )

        print("=" * 70)

    # ------------------------------------------------------

    def run(self):

        print(
            "\nStarting Final "
            "KMeans + RBF Pipeline...\n"
        )

        # Load RBF representation

        self.load_rbf_data()

        # Targeted tuning

        self.train()

        # Make sure a valid winner exists

        if self.best_model is None:

            raise RuntimeError(

                "No valid 4-cluster KMeans "
                "solution was found."

            )

        # Save tuning experiments

        self.save_tuning_results()

        # Save final model

        self.save_model()

        # Save player → cluster mapping

        self.save_cluster_assignments()

        # Visualization

        self.plot_clusters()

        # Final report

        self.print_summary()


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    trainer = FinalKMeansTrainer()

    trainer.run()