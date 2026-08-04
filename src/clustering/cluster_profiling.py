"""
cluster_profiling.py
--------------------

Profiles the final KMeans clusters using the original
interpretable football-player features.

Purpose
-------
1. Load final player-cluster assignments
2. Calculate cluster sizes
3. Calculate mean and median feature values
4. Compare each cluster with the overall dataset
5. Calculate standardized cluster profiles
6. Identify strongest distinguishing features
7. Generate heatmap and bar charts
8. Save profiling results

IMPORTANT:
Cluster labels 0, 1, 2 and 3 are arbitrary.
We assign meaningful football names only AFTER profiling.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS_DIR


class ClusterProfiler:

    def __init__(self):

        # --------------------------------------------------
        # Input / output directories
        # --------------------------------------------------

        self.input_path = os.path.join(
            RESULTS_DIR,
            "final_kmeans",
            "player_cluster_assignments.csv"
        )

        self.output_dir = os.path.join(
            RESULTS_DIR,
            "final_kmeans",
            "cluster_profiling"
        )

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        # --------------------------------------------------
        # These will be detected from the dataset
        # --------------------------------------------------

        self.df = None
        self.feature_columns = None

    # ======================================================
    # LOAD DATA
    # ======================================================

    def load_data(self):

        print("\n" + "=" * 70)
        print("LOADING FINAL CLUSTER ASSIGNMENTS")
        print("=" * 70)

        if not os.path.exists(self.input_path):

            raise FileNotFoundError(
                f"\nCluster assignment file not found:\n"
                f"{self.input_path}"
            )

        self.df = pd.read_csv(
            self.input_path
        )

        if "cluster" not in self.df.columns:

            raise ValueError(
                "Column 'cluster' was not found in "
                "player_cluster_assignments.csv"
            )

        print(
            f"\nDataset shape : {self.df.shape}"
        )

        print(
            f"Clusters found : "
            f"{sorted(self.df['cluster'].unique())}"
        )

    # ======================================================
    # IDENTIFY NUMERIC FEATURES
    # ======================================================

    def identify_features(self):

        """
        Automatically identify numeric columns that can
        be used for profiling.

        Excludes:
            cluster
            IDs
            obvious index columns
        """

        numeric_columns = (
            self.df
            .select_dtypes(
                include=[np.number]
            )
            .columns
            .tolist()
        )

        excluded_columns = {
            "cluster",
            "id",
            "player_id",
            "sofifa_id",
            "unnamed: 0",
            "index"
        }

        self.feature_columns = [

            column

            for column in numeric_columns

            if column.lower()
            not in excluded_columns

        ]

        if not self.feature_columns:

            raise ValueError(
                "No numeric profiling features found."
            )

        print("\nFeatures used for profiling:")
        print("-" * 40)

        for feature in self.feature_columns:

            print(feature)

    # ======================================================
    # CLUSTER SIZE
    # ======================================================

    def cluster_sizes(self):

        print("\n")
        print("=" * 70)
        print("CLUSTER DISTRIBUTION")
        print("=" * 70)

        counts = (
            self.df["cluster"]
            .value_counts()
            .sort_index()
        )

        percentages = (
            counts /
            len(self.df)
            * 100
        )

        distribution = pd.DataFrame({

            "Players":
                counts,

            "Percentage":
                percentages

        })

        distribution.index.name = (
            "Cluster"
        )

        print(
            "\n" +
            distribution.to_string()
        )

        output_path = os.path.join(

            self.output_dir,

            "cluster_sizes.csv"

        )

        distribution.to_csv(
            output_path
        )

        print(
            f"\nSaved -> {output_path}"
        )

    # ======================================================
    # MEAN PROFILE
    # ======================================================

    def calculate_means(self):

        self.cluster_means = (

            self.df
            .groupby("cluster")[
                self.feature_columns
            ]
            .mean()

        )

        output_path = os.path.join(

            self.output_dir,

            "cluster_feature_means.csv"

        )

        self.cluster_means.to_csv(
            output_path
        )

        print(
            f"\nCluster means saved -> "
            f"{output_path}"
        )

    # ======================================================
    # MEDIAN PROFILE
    # ======================================================

    def calculate_medians(self):

        self.cluster_medians = (

            self.df
            .groupby("cluster")[
                self.feature_columns
            ]
            .median()

        )

        output_path = os.path.join(

            self.output_dir,

            "cluster_feature_medians.csv"

        )

        self.cluster_medians.to_csv(
            output_path
        )

        print(
            f"Cluster medians saved -> "
            f"{output_path}"
        )

    # ======================================================
    # OVERALL DATASET MEANS
    # ======================================================

    def calculate_overall_means(self):

        self.overall_means = (

            self.df[
                self.feature_columns
            ]
            .mean()

        )

        output_path = os.path.join(

            self.output_dir,

            "overall_feature_means.csv"

        )

        self.overall_means.to_csv(
            output_path,
            header=["Overall Mean"]
        )

        print(
            f"Overall means saved -> "
            f"{output_path}"
        )

    # ======================================================
    # STANDARDIZED PROFILE
    # ======================================================

    def calculate_standardized_profiles(self):

        """
        Standardize cluster means relative to the entire
        dataset.

        Positive value:
            cluster is above dataset average.

        Negative value:
            cluster is below dataset average.

        Example:
            +1.5 = strongly above average
            -1.2 = strongly below average
        """

        overall_mean = (

            self.df[
                self.feature_columns
            ]
            .mean()

        )

        overall_std = (

            self.df[
                self.feature_columns
            ]
            .std()

        )

        # Prevent division by zero

        overall_std = (
            overall_std.replace(
                0,
                np.nan
            )
        )

        self.standardized_profiles = (

            (
                self.cluster_means
                -
                overall_mean
            )

            /

            overall_std

        )

        output_path = os.path.join(

            self.output_dir,

            "standardized_cluster_profiles.csv"

        )

        self.standardized_profiles.to_csv(
            output_path
        )

        print(
            f"Standardized profiles saved -> "
            f"{output_path}"
        )

    # ======================================================
    # TOP DISTINGUISHING FEATURES
    # ======================================================

    def top_features(self, top_n=5):

        """
        Find features that distinguish each cluster most
        strongly from the overall dataset.

        Uses absolute standardized difference.
        """

        rows = []

        print("\n")
        print("=" * 70)
        print("TOP DISTINGUISHING FEATURES")
        print("=" * 70)

        for cluster in (
            self.standardized_profiles.index
        ):

            profile = (
                self.standardized_profiles
                .loc[cluster]
            )

            strongest = (

                profile
                .abs()
                .sort_values(
                    ascending=False
                )
                .head(top_n)
            )

            print(
                f"\nCluster {cluster}"
            )

            print("-" * 40)

            for feature in strongest.index:

                value = profile[
                    feature
                ]

                if value > 0:

                    direction = (
                        "Above Average"
                    )

                else:

                    direction = (
                        "Below Average"
                    )

                print(
                    f"{feature:<30} "
                    f"{value:>8.3f} "
                    f"({direction})"
                )

                rows.append({

                    "Cluster":
                        cluster,

                    "Feature":
                        feature,

                    "Standardized Difference":
                        value,

                    "Direction":
                        direction

                })

        top_features_df = pd.DataFrame(
            rows
        )

        output_path = os.path.join(

            self.output_dir,

            "top_distinguishing_features.csv"

        )

        top_features_df.to_csv(

            output_path,

            index=False

        )

        print(
            f"\nTop features saved -> "
            f"{output_path}"
        )

    # ======================================================
    # FEATURE RANGE SUMMARY
    # ======================================================

    def calculate_feature_summary(self):

        """
        Mean, median and standard deviation for every
        cluster-feature combination.
        """

        summary = (

            self.df
            .groupby("cluster")[
                self.feature_columns
            ]
            .agg([
                "mean",
                "median",
                "std"
            ])

        )

        output_path = os.path.join(

            self.output_dir,

            "cluster_feature_summary.csv"

        )

        summary.to_csv(
            output_path
        )

        print(
            f"Feature summary saved -> "
            f"{output_path}"
        )

    # ======================================================
    # HEATMAP
    # ======================================================

    def plot_heatmap(self):

        """
        Plot standardized feature profiles without
        requiring seaborn.
        """

        data = (
            self.standardized_profiles
            .T
        )

        # Skip heatmap if there are too many features

        if len(data) > 40:

            print(
                "\nHeatmap skipped because more than "
                "40 numeric features were detected."
            )

            return

        fig, ax = plt.subplots(
            figsize=(
                10,
                max(
                    6,
                    len(data) * 0.35
                )
            )
        )

        image = ax.imshow(
            data.values,
            aspect="auto",
            cmap="coolwarm"
        )

        ax.set_xticks(
            range(
                len(data.columns)
            )
        )

        ax.set_xticklabels(

            [
                f"Cluster {cluster}"
                for cluster
                in data.columns
            ]

        )

        ax.set_yticks(
            range(
                len(data.index)
            )
        )

        ax.set_yticklabels(
            data.index
        )

        ax.set_title(
            "Standardized Football Player Cluster Profiles"
        )

        ax.set_xlabel(
            "Cluster"
        )

        ax.set_ylabel(
            "Feature"
        )

        plt.colorbar(
            image,
            ax=ax,
            label="Standardized Difference"
        )

        plt.tight_layout()

        output_path = os.path.join(

            self.output_dir,

            "cluster_profile_heatmap.png"

        )

        plt.savefig(

            output_path,

            dpi=300,

            bbox_inches="tight"

        )

        plt.close()

        print(
            f"Profile heatmap saved -> "
            f"{output_path}"
        )

    # ======================================================
    # CLUSTER PROFILE BAR CHARTS
    # ======================================================

    def plot_cluster_profiles(self):

        """
        Create one standardized bar chart per cluster.
        """

        for cluster in (
            self.standardized_profiles.index
        ):

            profile = (

                self.standardized_profiles
                .loc[cluster]
                .sort_values()

            )

            # Avoid unreadable plots if there are
            # extremely many numeric columns.

            if len(profile) > 20:

                strongest_features = (

                    profile
                    .abs()
                    .sort_values(
                        ascending=False
                    )
                    .head(15)
                    .index

                )

                profile = (

                    profile[
                        strongest_features
                    ]
                    .sort_values()

                )

            plt.figure(
                figsize=(10, 7)
            )

            profile.plot(
                kind="barh"
            )

            plt.axvline(
                x=0,
                linewidth=1
            )

            plt.xlabel(
                "Standardized Difference "
                "from Dataset Average"
            )

            plt.ylabel(
                "Feature"
            )

            plt.title(
                f"Cluster {cluster} Feature Profile"
            )

            plt.tight_layout()

            output_path = os.path.join(

                self.output_dir,

                f"cluster_{cluster}_profile.png"

            )

            plt.savefig(

                output_path,

                dpi=300,

                bbox_inches="tight"

            )

            plt.close()

            print(
                f"Cluster {cluster} profile plot "
                f"saved -> {output_path}"
            )

    # ======================================================
    # PRINT MEAN TABLE
    # ======================================================

    def print_profiles(self):

        print("\n")
        print("=" * 70)
        print("CLUSTER FEATURE MEANS")
        print("=" * 70)

        print(
            "\n" +
            self.cluster_means
            .round(3)
            .to_string()
        )

        print("\n")
        print("=" * 70)
        print("STANDARDIZED CLUSTER PROFILES")
        print("=" * 70)

        print(
            "\n" +
            self.standardized_profiles
            .round(3)
            .to_string()
        )

    # ======================================================
    # RUN
    # ======================================================

    def run(self):

        print(
            "\nStarting Cluster Profiling...\n"
        )

        self.load_data()

        self.identify_features()

        self.cluster_sizes()

        self.calculate_means()

        self.calculate_medians()

        self.calculate_overall_means()

        self.calculate_standardized_profiles()

        self.calculate_feature_summary()

        self.top_features(
            top_n=5
        )

        self.plot_heatmap()

        self.plot_cluster_profiles()

        self.print_profiles()

        print("\n")
        print("=" * 70)
        print("CLUSTER PROFILING COMPLETED")
        print("=" * 70)


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    profiler = ClusterProfiler()

    profiler.run()