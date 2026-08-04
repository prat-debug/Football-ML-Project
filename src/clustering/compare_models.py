"""
compare_models.py
-----------------

Final clustering model comparison and validation.

Purpose
-------
Compare KMeans, GMM, Hierarchical and DBSCAN across:

1. Linear PCA
2. Kernel PCA - RBF
3. Kernel PCA - Polynomial
4. Kernel PCA - Sigmoid
5. Kernel PCA - Cosine

Important:
A model is NOT selected only because it has a high
Silhouette Score.

A valid final clustering solution must:

1. Produce exactly 4 clusters
2. Avoid extremely tiny clusters
3. Avoid one cluster dominating almost all players
4. Consider DBSCAN noise
5. Then compare clustering evaluation metrics

Author:
Team Clustering Module
"""

import os
import numpy as np
import pandas as pd

from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering
)

from sklearn.mixture import GaussianMixture

from linear_pca import get_linear_pca_data
from kernel_pca import get_kernel_pca_data
from evaluation import ClusterEvaluator

from config import (
    RANDOM_STATE,
    RESULTS_DIR
)


# ==========================================================
# FINAL MODEL VALIDATOR
# ==========================================================

class FinalModelValidator:

    def __init__(self):

        # --------------------------------------------------
        # Project requirement
        # --------------------------------------------------

        self.target_clusters = 4

        # --------------------------------------------------
        # Practical cluster validation thresholds
        #
        # These are validation rules for your segmentation
        # objective, not universal clustering rules.
        # --------------------------------------------------

        self.min_cluster_percentage = 1.0

        self.max_cluster_percentage = 85.0

        self.max_dbscan_noise_percentage = 20.0

        # --------------------------------------------------

        self.datasets = {}

        self.results = []

        self.distributions = []

    # ======================================================
    # LOAD PCA REPRESENTATIONS
    # ======================================================

    def load_datasets(self):

        print("\n" + "=" * 70)
        print("LOADING PCA REPRESENTATIONS")
        print("=" * 70)

        # --------------------------------------------------
        # Linear PCA
        # --------------------------------------------------

        X_linear, _, _ = (
            get_linear_pca_data()
        )

        self.datasets[
            "Linear PCA"
        ] = X_linear

        print(
            f"Linear PCA loaded : "
            f"{X_linear.shape}"
        )

        # --------------------------------------------------
        # Kernel PCA
        # --------------------------------------------------

        transformed, _, _ = (
            get_kernel_pca_data()
        )

        for kernel, X_kernel in (
            transformed.items()
        ):

            name = (
                f"Kernel PCA ({kernel})"
            )

            self.datasets[
                name
            ] = X_kernel

            print(
                f"{name} loaded : "
                f"{X_kernel.shape}"
            )

    # ======================================================
    # CLUSTER DISTRIBUTION
    # ======================================================

    def get_distribution(
        self,
        algorithm,
        representation,
        labels
    ):

        labels = np.asarray(labels)

        total = len(labels)

        distribution = []

        for label in np.unique(labels):

            count = int(
                np.sum(labels == label)
            )

            percentage = (
                count / total
            ) * 100

            if label == -1:

                cluster_name = "Noise"

            else:

                cluster_name = (
                    f"Cluster {label}"
                )

            row = {

                "Algorithm":
                    algorithm,

                "Representation":
                    representation,

                "Cluster":
                    cluster_name,

                "Cluster Label":
                    int(label),

                "Players":
                    count,

                "Percentage":
                    percentage

            }

            distribution.append(row)

            self.distributions.append(
                row
            )

        return distribution

    # ======================================================
    # VALIDATE DISTRIBUTION
    # ======================================================

    def validate_distribution(
        self,
        algorithm,
        distribution
    ):

        valid_clusters = [

            row

            for row in distribution

            if row["Cluster Label"] != -1

        ]

        # --------------------------------------------------
        # Exactly 4 clusters
        # --------------------------------------------------

        if (
            len(valid_clusters)
            != self.target_clusters
        ):

            return (
                False,
                "Not exactly 4 clusters"
            )

        percentages = [

            row["Percentage"]

            for row in valid_clusters

        ]

        smallest = min(
            percentages
        )

        largest = max(
            percentages
        )

        # --------------------------------------------------
        # Tiny cluster check
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Dominant cluster check
        # --------------------------------------------------

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

        # --------------------------------------------------
        # DBSCAN noise validation
        # --------------------------------------------------

        if algorithm == "DBSCAN":

            noise_rows = [

                row

                for row in distribution

                if row["Cluster Label"] == -1

            ]

            if noise_rows:

                noise_percentage = (
                    noise_rows[0][
                        "Percentage"
                    ]
                )

                if (
                    noise_percentage
                    >
                    self.max_dbscan_noise_percentage
                ):

                    return (

                        False,

                        (
                            "Too much DBSCAN noise "
                            f"({noise_percentage:.2f}%)"
                        )

                    )

        return (
            True,
            "Valid"
        )

    # ======================================================
    # EVALUATE CANDIDATE
    # ======================================================

    def evaluate_candidate(
        self,
        algorithm,
        representation,
        X,
        labels,
        params
    ):

        labels = np.asarray(labels)

        # --------------------------------------------------
        # Distribution
        # --------------------------------------------------

        distribution = (
            self.get_distribution(

                algorithm,

                representation,

                labels

            )
        )

        # --------------------------------------------------
        # Practical validation
        # --------------------------------------------------

        valid, reason = (
            self.validate_distribution(

                algorithm,

                distribution

            )
        )

        # --------------------------------------------------
        # Number of clusters
        # --------------------------------------------------

        cluster_labels = set(
            labels
        )

        cluster_labels.discard(-1)

        number_clusters = len(
            cluster_labels
        )

        # --------------------------------------------------
        # Noise
        # --------------------------------------------------

        noise_points = int(
            np.sum(labels == -1)
        )

        noise_percentage = (

            noise_points /
            len(labels)

        ) * 100

        # --------------------------------------------------
        # Evaluation metrics
        # --------------------------------------------------

        if number_clusters >= 2:

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
                    number_clusters

            }

        valid_cluster_rows = [

            row

            for row in distribution

            if row["Cluster Label"] != -1

        ]

        if valid_cluster_rows:

            smallest_cluster = min(

                row["Percentage"]

                for row
                in valid_cluster_rows

            )

            largest_cluster = max(

                row["Percentage"]

                for row
                in valid_cluster_rows

            )

        else:

            smallest_cluster = 0

            largest_cluster = 0

        # --------------------------------------------------
        # Save result
        # --------------------------------------------------

        result = {

            "Algorithm":
                algorithm,

            "Representation":
                representation,

            "Silhouette Score":
                metrics[
                    "Silhouette Score"
                ],

            "Davies-Bouldin Index":
                metrics[
                    "Davies-Bouldin Index"
                ],

            "Calinski-Harabasz Score":
                metrics[
                    "Calinski-Harabasz Score"
                ],

            "Number of Clusters":
                number_clusters,

            "Smallest Cluster %":
                smallest_cluster,

            "Largest Cluster %":
                largest_cluster,

            "Noise Points":
                noise_points,

            "Noise Percentage":
                noise_percentage,

            "Valid Segmentation":
                valid,

            "Validation Reason":
                reason,

            "Parameters":
                str(params)

        }

        self.results.append(
            result
        )

        # --------------------------------------------------
        # Print result
        # --------------------------------------------------

        print(
            f"\n{algorithm} | "
            f"{representation}"
        )

        print(
            f"Clusters       : "
            f"{number_clusters}"
        )

        print(
            f"Smallest       : "
            f"{smallest_cluster:.2f}%"
        )

        print(
            f"Largest        : "
            f"{largest_cluster:.2f}%"
        )

        if algorithm == "DBSCAN":

            print(
                f"Noise          : "
                f"{noise_percentage:.2f}%"
            )

        print(
            f"Silhouette     : "
            f"{metrics['Silhouette Score']}"
        )

        print(
            f"Valid          : "
            f"{valid}"
        )

        print(
            f"Reason         : "
            f"{reason}"
        )

    # ======================================================
    # KMEANS
    # ======================================================

    def evaluate_kmeans(self):

        print("\n")
        print("=" * 70)
        print("VALIDATING KMEANS")
        print("=" * 70)

        for representation, X in (
            self.datasets.items()
        ):

            model = KMeans(

                n_clusters=4,

                init="k-means++",

                n_init=20,

                max_iter=300,

                algorithm="lloyd",

                random_state=RANDOM_STATE

            )

            labels = model.fit_predict(
                X
            )

            params = {

                "n_clusters": 4,

                "init":
                    "k-means++",

                "n_init": 20,

                "max_iter": 300,

                "algorithm":
                    "lloyd"

            }

            self.evaluate_candidate(

                "KMeans",

                representation,

                X,

                labels,

                params

            )

    # ======================================================
    # GMM
    # ======================================================

    def evaluate_gmm(self):

        print("\n")
        print("=" * 70)
        print("VALIDATING GMM")
        print("=" * 70)

        for representation, X in (
            self.datasets.items()
        ):

            model = GaussianMixture(

                n_components=4,

                covariance_type="tied",

                max_iter=200,

                init_params="kmeans",

                reg_covar=0.001,

                random_state=RANDOM_STATE

            )

            labels = model.fit_predict(
                X
            )

            params = {

                "n_components": 4,

                "covariance_type":
                    "tied",

                "max_iter": 200,

                "init_params":
                    "kmeans",

                "reg_covar":
                    0.001

            }

            self.evaluate_candidate(

                "GMM",

                representation,

                X,

                labels,

                params

            )

    # ======================================================
    # HIERARCHICAL
    # ======================================================

    def create_hierarchical(self):

        try:

            return (
                AgglomerativeClustering(

                    n_clusters=4,

                    linkage="complete",

                    metric="manhattan"

                )
            )

        except TypeError:

            return (
                AgglomerativeClustering(

                    n_clusters=4,

                    linkage="complete",

                    affinity="manhattan"

                )
            )

    # ------------------------------------------------------

    def evaluate_hierarchical(self):

        print("\n")
        print("=" * 70)
        print("VALIDATING HIERARCHICAL")
        print("=" * 70)

        for representation, X in (
            self.datasets.items()
        ):

            model = (
                self.create_hierarchical()
            )

            labels = model.fit_predict(
                X
            )

            params = {

                "n_clusters": 4,

                "linkage":
                    "complete",

                "metric":
                    "manhattan"

            }

            self.evaluate_candidate(

                "Hierarchical",

                representation,

                X,

                labels,

                params

            )

    # ======================================================
    # DBSCAN
    # ======================================================

    def evaluate_dbscan(self):

        print("\n")
        print("=" * 70)
        print("VALIDATING DBSCAN")
        print("=" * 70)

        # --------------------------------------------------
        # Use a small targeted grid.
        #
        # DBSCAN cannot directly request 4 clusters.
        # --------------------------------------------------

        eps_values = [

            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            1.0

        ]

        min_samples_values = [

            5,
            10,
            15

        ]

        for representation, X in (
            self.datasets.items()
        ):

            best_candidate = None

            for eps in eps_values:

                for min_samples in (
                    min_samples_values
                ):

                    model = DBSCAN(

                        eps=eps,

                        min_samples=
                            min_samples,

                        metric="euclidean",

                        algorithm="auto",

                        n_jobs=-1

                    )

                    labels = (
                        model.fit_predict(
                            X
                        )
                    )

                    unique_clusters = set(
                        labels
                    )

                    unique_clusters.discard(
                        -1
                    )

                    # ----------------------------------
                    # Only interested in 4 clusters
                    # ----------------------------------

                    if (
                        len(unique_clusters)
                        != 4
                    ):

                        continue

                    distribution = []

                    for label in np.unique(
                        labels
                    ):

                        count = int(
                            np.sum(
                                labels == label
                            )
                        )

                        percentage = (
                            count /
                            len(labels)
                        ) * 100

                        distribution.append({
                            "Cluster Label": int(label),
                            "Players": count,
                            "Percentage": percentage
                        })

                    valid, reason = (
                        self.validate_distribution(

                            "DBSCAN",

                            distribution

                        )
                    )

                    # ----------------------------------
                    # Evaluate only 4-cluster candidates
                    # ----------------------------------

                    metrics = (
                        ClusterEvaluator.evaluate(
                            X,
                            labels
                        )
                    )

                    score = metrics[
                        "Silhouette Score"
                    ]

                    candidate = {

                        "labels":
                            labels,

                        "params": {

                            "eps":
                                eps,

                            "min_samples":
                                min_samples,

                            "metric":
                                "euclidean"

                        },

                        "score":
                            score,

                        "valid":
                            valid

                    }

                    # ----------------------------------
                    # Prefer valid segmentation
                    # ----------------------------------

                    if best_candidate is None:

                        best_candidate = (
                            candidate
                        )

                    elif (
                        candidate["valid"]
                        and
                        not best_candidate[
                            "valid"
                        ]
                    ):

                        best_candidate = (
                            candidate
                        )

                    elif (

                        candidate["valid"]
                        ==
                        best_candidate[
                            "valid"
                        ]

                        and

                        candidate["score"]
                        >
                        best_candidate[
                            "score"
                        ]

                    ):

                        best_candidate = (
                            candidate
                        )

            # ------------------------------------------------
            # Evaluate best DBSCAN candidate
            # ------------------------------------------------

            if best_candidate is None:

                print(
                    f"\nDBSCAN | "
                    f"{representation}"
                )

                print(
                    "No 4-cluster "
                    "solution found."
                )

                continue

            self.evaluate_candidate(

                "DBSCAN",

                representation,

                X,

                best_candidate[
                    "labels"
                ],

                best_candidate[
                    "params"
                ]

            )

    # ======================================================
    # SAVE RESULTS
    # ======================================================

    def save_results(self):

        output_dir = os.path.join(

            RESULTS_DIR,

            "model_comparison"

        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        # --------------------------------------------------
        # Comparison DataFrame
        # --------------------------------------------------

        comparison_df = pd.DataFrame(
            self.results
        )

        comparison_path = os.path.join(

            output_dir,

            "representation_comparison.csv"

        )

        comparison_df.to_csv(

            comparison_path,

            index=False

        )

        print(f"\nComparison results saved to: {comparison_path}")

        # --------------------------------------------------
        # Distribution DataFrame
        # --------------------------------------------------

        distributions_df = pd.DataFrame(
            self.distributions
        )

        distributions_path = os.path.join(

            output_dir,

            "cluster_distributions.csv"

        )

        distributions_df.to_csv(

            distributions_path,

            index=False

        )

        print(f"Cluster distributions saved to: {distributions_path}")

        # --------------------------------------------------
        # Summary of valid candidates
        # --------------------------------------------------

        print("\n" + "=" * 70)
        print("VALIDATED CANDIDATES SUMMARY")
        print("=" * 70)

        valid_df = comparison_df[
            comparison_df["Valid Segmentation"] == True
        ].sort_values(
            by="Silhouette Score",
            ascending=False
        )

        if valid_df.empty:
            print("No algorithms produced a valid segmentation matching all rules.")
        else:
            print(
                valid_df[
                    [
                        "Algorithm",
                        "Representation",
                        "Silhouette Score",
                        "Davies-Bouldin Index",
                        "Calinski-Harabasz Score",
                        "Smallest Cluster %",
                        "Largest Cluster %"
                    ]
                ].to_string(index=False)
            )

    # ======================================================
    # RUN PIPELINE
    # ======================================================

    def run(self):

        self.load_datasets()

        self.evaluate_kmeans()

        self.evaluate_gmm()

        self.evaluate_hierarchical()

        self.evaluate_dbscan()

        self.save_results()


# ==========================================================
# MAIN EXECUTION
# ==========================================================

if __name__ == "__main__":

    validator = FinalModelValidator()

    validator.run()