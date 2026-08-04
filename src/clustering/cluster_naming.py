"""
cluster_naming.py
-----------------

Final cluster naming stage for the football player
clustering pipeline.

Final mapping:
    Cluster 0 -> Experienced Defenders
    Cluster 1 -> Elite Forwards
    Cluster 2 -> Young Talents
    Cluster 3 -> Defensive Specialists

Input:
    results/clustering/final_kmeans/
    player_cluster_assignments.csv

Outputs:
    1. final_named_player_clusters.csv
    2. cluster_name_mapping.csv
    3. named_cluster_summary.csv

Author:
Team Clustering Module
"""

import os
import pandas as pd

from config import RESULTS_DIR


class ClusterNaming:

    def __init__(self):

        # ==================================================
        # INPUT
        # ==================================================

        self.input_path = os.path.join(
            RESULTS_DIR,
            "final_kmeans",
            "player_cluster_assignments.csv"
        )

        # ==================================================
        # OUTPUT
        # ==================================================

        self.output_dir = os.path.join(
            RESULTS_DIR,
            "final_kmeans",
            "cluster_naming"
        )

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )

        # ==================================================
        # FINAL CLUSTER MAPPING
        # ==================================================

        self.cluster_mapping = {

            0: "Experienced Defenders",

            1: "Elite Forwards",

            2: "Young Talents",

            3: "Defensive Specialists"

        }

        self.df = None

    # ======================================================
    # LOAD DATA
    # ======================================================

    def load_data(self):

        print("\n" + "=" * 70)
        print("LOADING FINAL KMEANS CLUSTERS")
        print("=" * 70)

        if not os.path.exists(
            self.input_path
        ):

            raise FileNotFoundError(
                f"\nInput file not found:\n"
                f"{self.input_path}"
            )

        self.df = pd.read_csv(
            self.input_path
        )

        if "cluster" not in self.df.columns:

            raise ValueError(
                "Column 'cluster' not found "
                "in player cluster assignments."
            )

        print(
            f"\nDataset Shape : "
            f"{self.df.shape}"
        )

        print(
            "Clusters Found :",
            sorted(
                self.df[
                    "cluster"
                ].unique()
            )
        )

    # ======================================================
    # VALIDATE CLUSTERS
    # ======================================================

    def validate_clusters(self):

        print("\n" + "=" * 70)
        print("VALIDATING CLUSTER LABELS")
        print("=" * 70)

        actual_clusters = set(
            self.df[
                "cluster"
            ].unique()
        )

        expected_clusters = set(
            self.cluster_mapping.keys()
        )

        # --------------------------------------------------
        # Check for unexpected clusters
        # --------------------------------------------------

        unexpected = (
            actual_clusters
            -
            expected_clusters
        )

        if unexpected:

            raise ValueError(
                f"Unexpected cluster labels found: "
                f"{unexpected}"
            )

        # --------------------------------------------------
        # Check for missing clusters
        # --------------------------------------------------

        missing = (
            expected_clusters
            -
            actual_clusters
        )

        if missing:

            raise ValueError(
                f"Expected clusters missing: "
                f"{missing}"
            )

        print(
            "\nAll four expected clusters "
            "are present."
        )

    # ======================================================
    # ASSIGN CLUSTER NAMES
    # ======================================================

    def assign_names(self):

        print("\n" + "=" * 70)
        print("ASSIGNING FINAL CLUSTER NAMES")
        print("=" * 70)

        self.df[
            "cluster_name"
        ] = (

            self.df[
                "cluster"
            ]
            .map(
                self.cluster_mapping
            )

        )

        # --------------------------------------------------
        # Safety check
        # --------------------------------------------------

        if (
            self.df[
                "cluster_name"
            ]
            .isna()
            .any()
        ):

            invalid = (

                self.df.loc[
                    self.df[
                        "cluster_name"
                    ].isna(),
                    "cluster"
                ]
                .unique()

            )

            raise ValueError(
                "Some clusters could not be named: "
                f"{invalid}"
            )

        print("\nFinal Mapping")
        print("-" * 45)

        for cluster, name in (
            self.cluster_mapping.items()
        ):

            print(
                f"Cluster {cluster:<3} "
                f"-> {name}"
            )

    # ======================================================
    # POSITION VALIDATION SUMMARY
    # ======================================================

    def position_summary(self):

        if (
            "position_group"
            not in self.df.columns
        ):

            print(
                "\nposition_group column not found. "
                "Skipping position summary."
            )

            return

        print("\n" + "=" * 70)
        print("POSITION DISTRIBUTION BY FINAL CLUSTER")
        print("=" * 70)

        position_table = pd.crosstab(

            self.df[
                "cluster_name"
            ],

            self.df[
                "position_group"
            ],

            normalize="index"

        ) * 100

        position_table = (
            position_table.round(2)
        )

        print(
            "\n" +
            position_table.to_string()
        )

        output_path = os.path.join(

            self.output_dir,

            "named_cluster_position_distribution.csv"

        )

        position_table.to_csv(
            output_path
        )

        print(
            f"\nSaved -> "
            f"{output_path}"
        )

    # ======================================================
    # CREATE CLUSTER SUMMARY
    # ======================================================

    def create_summary(self):

        print("\n" + "=" * 70)
        print("FINAL CLUSTER SUMMARY")
        print("=" * 70)

        counts = (

            self.df[
                "cluster_name"
            ]
            .value_counts()

        )

        percentages = (

            counts /
            len(self.df)

        ) * 100

        summary = pd.DataFrame({

            "Players":
                counts,

            "Percentage":
                percentages

        })

        summary.index.name = (
            "Cluster Name"
        )

        # --------------------------------------------------
        # Put cluster number back into summary
        # --------------------------------------------------

        reverse_mapping = {

            name: cluster

            for cluster, name
            in self.cluster_mapping.items()

        }

        summary[
            "Cluster"
        ] = (

            summary.index
            .map(
                reverse_mapping
            )

        )

        summary = (

            summary
            .reset_index()

        )

        summary = summary[

            [
                "Cluster",
                "Cluster Name",
                "Players",
                "Percentage"
            ]

        ]

        summary = (

            summary
            .sort_values(
                "Cluster"
            )

        )

        print(
            "\n" +
            summary
            .round(2)
            .to_string(
                index=False
            )
        )

        output_path = os.path.join(

            self.output_dir,

            "named_cluster_summary.csv"

        )

        summary.to_csv(

            output_path,

            index=False

        )

        print(
            f"\nSaved -> "
            f"{output_path}"
        )

        self.summary = summary

    # ======================================================
    # SAVE CLUSTER MAPPING
    # ======================================================

    def save_mapping(self):

        mapping_df = pd.DataFrame({

            "cluster":
                list(
                    self.cluster_mapping.keys()
                ),

            "cluster_name":
                list(
                    self.cluster_mapping.values()
                )

        })

        output_path = os.path.join(

            self.output_dir,

            "cluster_name_mapping.csv"

        )

        mapping_df.to_csv(

            output_path,

            index=False

        )

        print(
            f"\nCluster mapping saved -> "
            f"{output_path}"
        )

    # ======================================================
    # SAVE FINAL DATASET
    # ======================================================

    def save_final_dataset(self):

        output_path = os.path.join(

            self.output_dir,

            "final_named_player_clusters.csv"

        )

        self.df.to_csv(

            output_path,

            index=False

        )

        print(
            f"\nFinal named player dataset "
            f"saved ->\n{output_path}"
        )

    # ======================================================
    # PRINT SAMPLE PLAYERS
    # ======================================================

    def print_samples(self):

        print("\n" + "=" * 70)
        print("SAMPLE PLAYERS FROM EACH CLUSTER")
        print("=" * 70)

        # Find a useful player-name column if available

        possible_name_columns = [

            "player_name",
            "short_name",
            "long_name",
            "name",
            "player"

        ]

        name_column = None

        for column in (
            possible_name_columns
        ):

            if column in self.df.columns:

                name_column = column
                break

        for cluster, name in (
            self.cluster_mapping.items()
        ):

            cluster_df = self.df[

                self.df[
                    "cluster"
                ] == cluster

            ]

            print(
                f"\n{name}"
            )

            print("-" * 50)

            columns = []

            if name_column:

                columns.append(
                    name_column
                )

            if (
                "position_group"
                in cluster_df.columns
            ):

                columns.append(
                    "position_group"
                )

            # Add age if available

            if "age" in cluster_df.columns:

                columns.append(
                    "age"
                )

            if columns:

                print(

                    cluster_df[
                        columns
                    ]
                    .head(5)
                    .to_string(
                        index=False
                    )

                )

            else:

                print(
                    cluster_df
                    .head(5)
                    .to_string(
                        index=False
                    )
                )

    # ======================================================
    # RUN
    # ======================================================

    def run(self):

        print(
            "\nStarting Final "
            "Cluster Naming Pipeline...\n"
        )

        self.load_data()

        self.validate_clusters()

        self.assign_names()

        self.create_summary()

        self.position_summary()

        self.save_mapping()

        self.save_final_dataset()

        self.print_samples()

        print("\n")
        print("=" * 70)
        print("FINAL CLUSTER NAMING COMPLETED")
        print("=" * 70)

        print(
            "\nFinal clusters:"
        )

        for cluster, name in (
            self.cluster_mapping.items()
        ):

            print(
                f"  {cluster} -> {name}"
            )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    naming = ClusterNaming()

    naming.run()