"""
evaluation.py
-------------

Common evaluation functions for clustering algorithms.

Used by:
    - KMeans
    - DBSCAN
    - Gaussian Mixture
    - Hierarchical Clustering

Author : Team Clustering Module
"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)


class ClusterEvaluator:
    """
    Computes clustering evaluation metrics.
    """

    @staticmethod
    def evaluate(X, labels):
        """
        Evaluate clustering performance.

        Parameters
        ----------
        X : ndarray or DataFrame
            Feature matrix.

        labels : ndarray
            Cluster labels.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics.
        """

        labels = np.asarray(labels)

        # ------------------------------------------
        # Remove DBSCAN noise label (-1)
        # ------------------------------------------

        unique_labels = set(labels)

        if -1 in unique_labels:
            unique_labels.remove(-1)

        # ------------------------------------------
        # Need at least 2 clusters
        # ------------------------------------------

        if len(unique_labels) < 2:

            return {
                "Silhouette Score": np.nan,
                "Davies-Bouldin Index": np.nan,
                "Calinski-Harabasz Score": np.nan,
                "Number of Clusters": len(unique_labels)
            }

        # ------------------------------------------
        # Ignore noise points for evaluation
        # ------------------------------------------

        if -1 in labels:

            mask = labels != -1

            X_eval = X[mask]

            labels_eval = labels[mask]

        else:

            X_eval = X

            labels_eval = labels

        # ------------------------------------------
        # Compute metrics safely
        # ------------------------------------------

        try:

            metrics = {

                "Silhouette Score":
                    silhouette_score(X_eval, labels_eval),

                "Davies-Bouldin Index":
                    davies_bouldin_score(X_eval, labels_eval),

                "Calinski-Harabasz Score":
                    calinski_harabasz_score(X_eval, labels_eval),

                "Number of Clusters":
                    len(np.unique(labels_eval))

            }

        except Exception as e:

            print(f"Evaluation Warning: {e}")

            metrics = {

                "Silhouette Score": np.nan,

                "Davies-Bouldin Index": np.nan,

                "Calinski-Harabasz Score": np.nan,

                "Number of Clusters":
                    len(np.unique(labels_eval))

            }

        return metrics

    # ------------------------------------------------------

    @staticmethod
    def save_metrics(metrics, output_path):
        """
        Save evaluation metrics to CSV.
        """

        df = pd.DataFrame([metrics])

        df.to_csv(output_path, index=False)

        print(f"\nMetrics saved to:\n{output_path}")