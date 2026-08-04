"""
kernel_pca.py
-------------

Performs Kernel PCA using multiple kernels.

Kernels Tested
--------------
1. RBF
2. Polynomial
3. Sigmoid
4. Cosine

Author : Team Clustering Module
"""

import pandas as pd

from sklearn.decomposition import KernelPCA

from preprocessing import get_preprocessed_data

from config import (
    RANDOM_STATE,
    KERNELS,
    PCA_COMPONENTS,
    KERNEL_RESULTS_DIR
)


class KernelPCATransformer:

    def __init__(self):

        self.transformers = {}

    # -----------------------------------------------------

    def transform_all(self, X):

        transformed_data = {}

        for kernel in KERNELS:

            print(f"\nRunning Kernel PCA ({kernel})...")

            kpca = KernelPCA(

                n_components=PCA_COMPONENTS,

                kernel=kernel,

                fit_inverse_transform=False,

                random_state=RANDOM_STATE

            )

            X_kernel = kpca.fit_transform(X)

            transformed_data[kernel] = X_kernel

            self.transformers[kernel] = kpca

        return transformed_data

    # -----------------------------------------------------

    def save_component_shapes(self, transformed_data):

        rows = []

        for kernel, X in transformed_data.items():

            rows.append({

                "Kernel": kernel,

                "Samples": X.shape[0],

                "Components": X.shape[1]

            })

        df = pd.DataFrame(rows)

        output_path = KERNEL_RESULTS_DIR / "kernel_summary.csv"

        df.to_csv(output_path, index=False)

        print(f"\nKernel summary saved to:\n{output_path}")


# =======================================================

def get_kernel_pca_data():

    X_scaled, positions, df = get_preprocessed_data()

    transformer = KernelPCATransformer()

    transformed = transformer.transform_all(X_scaled)

    return transformed, positions, df


# =======================================================

if __name__ == "__main__":

    X_scaled, positions, df = get_preprocessed_data()

    transformer = KernelPCATransformer()

    transformed = transformer.transform_all(X_scaled)

    transformer.save_component_shapes(transformed)

    print("\nKernel PCA completed successfully.\n")

    for kernel, X in transformed.items():

        print(f"{kernel:10s} : {X.shape}")