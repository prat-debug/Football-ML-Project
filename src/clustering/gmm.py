"""
gmm.py
------

Gaussian Mixture Model (GMM) Clustering Module

Performs:
1. Linear PCA + GMM
2. Kernel PCA (RBF) + GMM
3. Kernel PCA (Poly) + GMM
4. Kernel PCA (Sigmoid) + GMM
5. Kernel PCA (Cosine) + GMM

Exactly 4 clusters are generated.
Performs hyperparameter tuning.

Evaluation:
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Score
- BIC
- AIC

Saves:
- Metrics CSV
- Cluster plots
- Best GMM model

Author:
Team Clustering Module
"""

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product
from sklearn.mixture import GaussianMixture

from linear_pca import get_linear_pca_data
from kernel_pca import get_kernel_pca_data
from evaluation import ClusterEvaluator

from config import (
    RANDOM_STATE,
    MODELS_DIR,
    LINEAR_RESULTS_DIR,
    KERNEL_RESULTS_DIR
)


class GMMTrainer:
    """
    Handles:
    - GMM hyperparameter tuning
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
        # GMM HYPERPARAMETER GRID
        # ==================================================
        self.param_grid = {
            # Project requirement: exactly 4 player clusters
            "n_components": [4],
            "covariance_type": [
                "full",
                "tied",
                "diag",
                "spherical"
            ],
            "max_iter": [
                200,
                500
            ],
            "init_params": [
                "kmeans",
                "random"
            ],
            "reg_covar": [
                1e-6,
                1e-4,
                1e-3
            ]
        }

    # -----------------------------------------------------

    def parameter_combinations(self):
        """
        Generate all GMM hyperparameter combinations.
        """
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

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
        Tune GMM on one PCA representation.
        """
        print("\n" + "=" * 70)
        print(f"Training GMM : {dataset_name}")
        print("=" * 70)

        best_model = None
        best_labels = None
        best_metrics = None
        best_params = None

        best_bic = None
        best_aic = None

        best_score = float("-inf")
        all_results = []

        # ==================================================
        # HYPERPARAMETER TUNING
        # ==================================================
        for params in self.parameter_combinations():
            try:
                model = GaussianMixture(
                    random_state=RANDOM_STATE,
                    **params
                )

                # Train model
                labels = model.fit_predict(X)

                # Evaluate clustering
                metrics = ClusterEvaluator.evaluate(
                    X,
                    labels
                )

                # GMM-specific metrics
                bic = model.bic(X)
                aic = model.aic(X)

                # Store experiment
                row = {
                    **params,
                    **metrics,
                    "BIC": bic,
                    "AIC": aic,
                    "Converged": model.converged_,
                    "Iterations": model.n_iter_
                }

                all_results.append(row)

                # Validate 4 clusters
                if metrics.get("Number of Clusters") != 4:
                    continue

                # Select best model
                score = metrics.get("Silhouette Score")

                if pd.isna(score):
                    continue

                if score > best_score:
                    best_score = score
                    best_model = model
                    best_labels = labels
                    best_metrics = metrics
                    best_params = params.copy()
                    best_bic = bic
                    best_aic = aic

            except Exception as error:
                print(
                    f"Skipping invalid combination "
                    f"{params}: {error}"
                )

        # ==================================================
        # SAVE ALL TUNING RESULTS
        # ==================================================
        results_df = pd.DataFrame(all_results)

        if not results_df.empty:
            results_df = results_df.sort_values(
                by="Silhouette Score",
                ascending=False,
                na_position="last"
            )

        os.makedirs(save_directory, exist_ok=True)

        csv_path = os.path.join(save_directory, "gmm_metrics.csv")
        results_df.to_csv(csv_path, index=False)

        print(f"\nMetrics saved -> {csv_path}")

        return {
            "model": best_model,
            "labels": best_labels,
            "metrics": best_metrics,
            "params": best_params,
            "bic": best_bic,
            "aic": best_aic,
            "results": results_df
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
        Plot GMM clusters using first two PCA components.
        """
        os.makedirs(save_directory, exist_ok=True)

        if hasattr(X, "values"):
            X_plot = X.values
        else:
            X_plot = X

        plt.figure(figsize=(8, 6))

        scatter = plt.scatter(
            X_plot[:, 0],
            X_plot[:, 1],
            c=labels,
            cmap="tab10",
            s=20,
            alpha=0.8
        )

        plt.title(title)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(scatter, label="Cluster")

        plot_path = os.path.join(save_directory, "gmm_clusters.png")

        plt.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        print(f"Cluster plot saved -> {plot_path}")

    # -----------------------------------------------------

    def train_linear_pca(self):
        """
        Train GMM using Linear PCA.
        """
        X_linear, _, _ = get_linear_pca_data()

        save_dir = os.path.join(
            LINEAR_RESULTS_DIR,
            "gmm"
        )

        linear_result = self.train_single_dataset(
            X_linear,
            "Linear PCA",
            save_dir
        )

        if linear_result["labels"] is not None:
            self.plot_clusters(
                X_linear,
                linear_result["labels"],
                save_dir,
                "GMM - Linear PCA"
            )
        else:
            print("\nNo valid 4-cluster GMM solution found for Linear PCA.")

        self.all_models["Linear PCA"] = linear_result

    # -----------------------------------------------------

        # -----------------------------------------------------

    def train_kernel_pca(self):
        """
        Train GMM using all Kernel PCA representations.
        """

        # get_kernel_pca_data() returns:
        # transformed, positions, df

        kernel_datasets, _, _ = get_kernel_pca_data()

        # kernel_datasets is now the dictionary:
        #
        # {
        #     "rbf": X_rbf,
        #     "poly": X_poly,
        #     "sigmoid": X_sigmoid,
        #     "cosine": X_cosine
        # }

        for kernel_name, X_kernel in kernel_datasets.items():

            save_dir = os.path.join(
                KERNEL_RESULTS_DIR,
                kernel_name,
                "gmm"
            )

            kernel_result = self.train_single_dataset(
                X_kernel,
                f"Kernel PCA ({kernel_name})",
                save_dir
            )

            if kernel_result["labels"] is not None:

                self.plot_clusters(
                    X_kernel,
                    kernel_result["labels"],
                    save_dir,
                    f"GMM - {kernel_name.upper()}"
                )

            else:

                print(
                    f"\nNo valid 4-cluster GMM solution "
                    f"found for Kernel PCA ({kernel_name})."
                )

            self.all_models[
                f"Kernel PCA ({kernel_name})"
            ] = kernel_result

    # -----------------------------------------------------

    def compare_models(self):
        """
        Compare the best 4-cluster GMM result from every PCA representation.
        """
        best_name = None
        best_score = float("-inf")

        print("\n")
        print("=" * 70)
        print("GMM MODEL COMPARISON")
        print("=" * 70)

        for dataset_name, result in self.all_models.items():
            metrics = result["metrics"]

            if metrics is None:
                print("\n" + "-" * 60)
                print(dataset_name)
                print("No valid 4-cluster GMM solution found.")
                continue

            score = metrics.get("Silhouette Score")

            if pd.isna(score):
                continue

            print("\n" + "-" * 60)
            print(dataset_name)

            for key, value in metrics.items():
                print(f"{key:<30}: {value}")

            print(f"{'BIC':<30}: {result['bic']}")
            print(f"{'AIC':<30}: {result['aic']}")
            print(f"{'Converged':<30}: {result['model'].converged_}")

            if score > best_score:
                best_score = score
                best_name = dataset_name

        if best_name is None:
            raise RuntimeError(
                "GMM could not produce a valid 4-cluster solution for any PCA representation."
            )

        self.best_dataset = best_name
        self.best_result = self.all_models[best_name]

    # -----------------------------------------------------

    def save_best_model(self):
        """
        Save overall best 4-cluster GMM model.
        """
        output_path = os.path.join(MODELS_DIR, "gmm_best.pkl")

        joblib.dump(self.best_result["model"], output_path)

        print("\n" + "=" * 70)
        print("Best GMM Model Saved")
        print("=" * 70)
        print(output_path)

    # -----------------------------------------------------

    def print_summary(self):
        """
        Print final GMM result.
        """
        print("\n")
        print("=" * 70)
        print("FINAL GMM RESULT")
        print("=" * 70)
        print(f"\nWinner : {self.best_dataset}\n")

        print("Best Hyperparameters")
        print("-" * 30)

        best_params = self.best_result["params"]
        for key, value in best_params.items():
            print(f"{key:<30}: {value}")

        print("\nEvaluation Metrics")
        print("-" * 30)

        metrics = self.best_result["metrics"]
        for key, value in metrics.items():
            print(f"{key:<30}: {value}")

        print("\nGMM Information")
        print("-" * 30)

        print(f"{'BIC':<30}: {self.best_result['bic']}")
        print(f"{'AIC':<30}: {self.best_result['aic']}")

        model = self.best_result["model"]
        print(f"{'Converged':<30}: {model.converged_}")
        print(f"{'Iterations':<30}: {model.n_iter_}")
        print("=" * 70)

    # -----------------------------------------------------

    def run(self):
        """
        Run complete 4-cluster GMM pipeline.
        """
        print("\nStarting GMM Pipeline...\n")

        # Linear PCA
        self.train_linear_pca()

        # All Kernel PCA representations
        self.train_kernel_pca()

        # Compare all representations
        self.compare_models()

        # Save winner
        self.save_best_model()

        # Final result
        self.print_summary()


# ==========================================================
# Main
# ==========================================================

if __name__ == "__main__":
    trainer = GMMTrainer()
    trainer.run()