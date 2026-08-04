"""
hierarchical.py
---------------

Hierarchical / Agglomerative Clustering Module

Performs:

1. Linear PCA + Hierarchical Clustering
2. Kernel PCA (RBF) + Hierarchical Clustering
3. Kernel PCA (Poly) + Hierarchical Clustering
4. Kernel PCA (Sigmoid) + Hierarchical Clustering
5. Kernel PCA (Cosine) + Hierarchical Clustering

Project requirement:
Exactly 4 player clusters.

Hyperparameters tuned:
- linkage
- metric

Evaluation:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score

Saves:
- Metrics CSV
- Cluster plots
- Best Hierarchical model

Author:
Team Clustering Module
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering

from linear_pca import get_linear_pca_data
from kernel_pca import get_kernel_pca_data
from evaluation import ClusterEvaluator

from config import (
    MODELS_DIR,
    LINEAR_RESULTS_DIR,
    KERNEL_RESULTS_DIR
)


class HierarchicalTrainer:
    """
    Handles:

    - Hierarchical clustering
    - Hyperparameter tuning
    - Linear PCA experiments
    - Kernel PCA experiments
    - Model comparison
    - Best model saving
    """

    def __init__(self):

        self.all_models = {}

        self.best_dataset = None
        self.best_result = None

        # --------------------------------------------------
        # Exactly 4 clusters required
        # --------------------------------------------------

        self.n_clusters = 4

        # --------------------------------------------------
        # Valid linkage + metric combinations
        #
        # IMPORTANT:
        # Ward linkage only supports Euclidean distance.
        # --------------------------------------------------

        self.param_grid = [

            {
                "linkage": "ward",
                "metric": "euclidean"
            },

            {
                "linkage": "complete",
                "metric": "euclidean"
            },

            {
                "linkage": "complete",
                "metric": "manhattan"
            },

            {
                "linkage": "complete",
                "metric": "cosine"
            },

            {
                "linkage": "average",
                "metric": "euclidean"
            },

            {
                "linkage": "average",
                "metric": "manhattan"
            },

            {
                "linkage": "average",
                "metric": "cosine"
            },

            {
                "linkage": "single",
                "metric": "euclidean"
            },

            {
                "linkage": "single",
                "metric": "manhattan"
            },

            {
                "linkage": "single",
                "metric": "cosine"
            }

        ]

    # -----------------------------------------------------

    def create_model(
        self,
        linkage,
        metric
    ):
        """
        Create AgglomerativeClustering model.

        Handles compatibility between newer and older
        scikit-learn versions.

        New sklearn versions use:
            metric=

        Older sklearn versions use:
            affinity=
        """

        try:

            model = AgglomerativeClustering(

                n_clusters=self.n_clusters,

                linkage=linkage,

                metric=metric

            )

        except TypeError:

            model = AgglomerativeClustering(

                n_clusters=self.n_clusters,

                linkage=linkage,

                affinity=metric

            )

        return model

    # -----------------------------------------------------

    def train_single_dataset(
        self,
        X,
        dataset_name,
        save_directory
    ):
        """
        Tune Hierarchical Clustering on one
        PCA representation.
        """

        print("\n" + "=" * 70)

        print(
            f"Training Hierarchical : {dataset_name}"
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

        for params in self.param_grid:

            try:

                model = self.create_model(

                    linkage=params["linkage"],

                    metric=params["metric"]

                )

                # ------------------------------------------
                # Train + generate labels
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
                # Store experiment
                # ------------------------------------------

                row = {

                    "n_clusters":
                        self.n_clusters,

                    "linkage":
                        params["linkage"],

                    "metric":
                        params["metric"],

                    **metrics

                }

                all_results.append(row)

                # ------------------------------------------
                # Safety check:
                # only accept exactly 4 clusters
                # ------------------------------------------

                if (
                    metrics["Number of Clusters"]
                    != self.n_clusters
                ):

                    continue

                # ------------------------------------------
                # Select using Silhouette Score
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
        # SAVE ALL RESULTS
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

            "hierarchical_metrics.csv"

        )

        results_df.to_csv(

            csv_path,

            index=False

        )

        print(
            f"\nMetrics saved -> {csv_path}"
        )

        # --------------------------------------------------
        # Return best result for this representation
        # --------------------------------------------------

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

            "hierarchical_clusters.png"

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
        Train Hierarchical Clustering using Linear PCA.
        """

        # Your current get_linear_pca_data()
        # returns 3 values.

        X_linear, _, _ = (
            get_linear_pca_data()
        )

        save_dir = os.path.join(

            LINEAR_RESULTS_DIR,

            "hierarchical"

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

                "Hierarchical - Linear PCA"

            )

        else:

            print(
                "\nNo valid 4-cluster Hierarchical "
                "solution found for Linear PCA."
            )

        self.all_models[
            "Linear PCA"
        ] = linear_result

    # -----------------------------------------------------

    def train_kernel_pca(self):
        """
        Train Hierarchical Clustering using all
        Kernel PCA representations.
        """

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

                "hierarchical"

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
                        "Hierarchical - "
                        f"{kernel_name.upper()}"
                    )

                )

            else:

                print(
                    f"\nNo valid 4-cluster "
                    f"Hierarchical solution found "
                    f"for Kernel PCA "
                    f"({kernel_name})."
                )

            self.all_models[
                f"Kernel PCA ({kernel_name})"
            ] = kernel_result

    # -----------------------------------------------------

    def compare_models(self):
        """
        Compare best Hierarchical result from
        every PCA representation.
        """

        best_name = None

        best_score = float("-inf")

        print("\n")

        print("=" * 70)

        print(
            "HIERARCHICAL MODEL COMPARISON"
        )

        print("=" * 70)

        for dataset_name, result in (
            self.all_models.items()
        ):

            metrics = result[
                "metrics"
            ]

            if metrics is None:

                print(
                    "\n" + "-" * 60
                )

                print(dataset_name)

                print(
                    "No valid 4-cluster solution."
                )

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
            # Print metrics
            # ------------------------------------------

            for key, value in (
                metrics.items()
            ):

                print(
                    f"{key:<30}: {value}"
                )

            # ------------------------------------------
            # Print hyperparameters
            # ------------------------------------------

            print(
                f"{'Linkage':<30}: "
                f"{result['params']['linkage']}"
            )

            print(
                f"{'Metric':<30}: "
                f"{result['params']['metric']}"
            )

            # ------------------------------------------
            # Compare
            # ------------------------------------------

            if score > best_score:

                best_score = score

                best_name = dataset_name

        # --------------------------------------------------
        # Ensure valid solution exists
        # --------------------------------------------------

        if best_name is None:

            raise RuntimeError(

                "Hierarchical Clustering could not "
                "produce a valid 4-cluster solution."

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
        Save overall best Hierarchical model.
        """

        output_path = os.path.join(

            MODELS_DIR,

            "hierarchical_best.pkl"

        )

        joblib.dump(

            self.best_result["model"],

            output_path

        )

        print(
            "\n" + "=" * 70
        )

        print(
            "Best Hierarchical Model Saved"
        )

        print("=" * 70)

        print(output_path)

    # -----------------------------------------------------

    def print_summary(self):
        """
        Print final Hierarchical result.
        """

        print("\n")

        print("=" * 70)

        print(
            "FINAL HIERARCHICAL RESULT"
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

        print(
            f"{'n_clusters':<30}: "
            f"{self.n_clusters}"
        )

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

        print("=" * 70)

    # -----------------------------------------------------

    def run(self):
        """
        Run complete Hierarchical pipeline.
        """

        print(
            "\nStarting Hierarchical "
            "Clustering Pipeline...\n"
        )

        # Linear PCA
        self.train_linear_pca()

        # Kernel PCA
        self.train_kernel_pca()

        # Compare all representations
        self.compare_models()

        # Save winner
        self.save_best_model()

        # Final summary
        self.print_summary()


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":

    trainer = HierarchicalTrainer()

    trainer.run()