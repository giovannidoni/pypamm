#!/usr/bin/env python
"""
Quick-Shift Clustering Example

This example demonstrates how to use the Quick-Shift clustering algorithm
from the PyPAMM library and compares it with hierarchical clustering.
Quick-Shift is a mode-seeking algorithm that finds clusters by shifting
points towards higher density regions.

The example uses a dataset defined in the example_config.yaml file and visualizes
the clustering results using the plot_clusters function, comparing Quick-Shift
with hierarchical clustering in a single image.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from data_generator import generate_dataset, load_config, plot_clusters
from matplotlib.gridspec import GridSpec
from sklearn.cluster import AgglomerativeClustering

from pypamm.density.kde import compute_kde
from pypamm.quick_shift import quick_shift_clustering

# Load configuration from YAML file
config = load_config("example_config")

# Get visualization settings
viz_config = config["visualization"]
output_dir = viz_config["output_dir"]
os.makedirs(output_dir, exist_ok=True)

this_config = config["examples"]["quick_shift_example"]
n_datasets = len(this_config["datasets"])
n_algorithms = len(this_config["algorithms"])

fig = plt.figure(figsize=(6 * n_datasets, 5 * n_algorithms))
gs = GridSpec(n_datasets, n_algorithms, figure=fig)

for i, dataset_name in enumerate(this_config["datasets"]):
    for j, algorithm_name in enumerate(this_config["algorithms"]):
        print(f"\nProcessing dataset: {dataset_name} with algorithm: {algorithm_name}")

        # Generate dataset
        X = generate_dataset(dataset_name, "data_config")

        # Create subplot
        ax = fig.add_subplot(gs[i, j])

        # Get algorithm parameters
        if algorithm_name == "hierarchical_clustering":
            # Apply hierarchical clustering
            hc_config = this_config["algorithms"][algorithm_name]

            hc = AgglomerativeClustering(linkage=hc_config["linkage"])
            labels = hc.fit_predict(X)
            n_found_clusters = len(np.unique(labels))

            title = f"Hierarchical Clustering ({hc_config['linkage']})\n{n_found_clusters} clusters"

        elif algorithm_name == "quick_shift":
            # Apply Quick-Shift clustering
            qs_config = this_config["algorithms"][algorithm_name]

            # Compute KDE for density estimation
            density = compute_kde(X, X, qs_config["bandwidth"])

            # Apply Quick-Shift clustering
            labels, centers = quick_shift_clustering(
                X, density, lambda_qs=qs_config["lambda_qs"], ngrid=qs_config["ngrid"], metric=qs_config["metric"]
            )
            n_found_clusters = len(np.unique(labels))

            title = (
                f"Quick-Shift Clustering\n{n_found_clusters} clusters "
                f"λ={qs_config['lambda_qs']}, max_dist={qs_config['max_dist']}"
            )
            print(f"Quick-Shift found {n_found_clusters} clusters")

        else:
            # Unknown algorithm
            print(f"Warning: Unknown algorithm '{algorithm_name}'")
            continue

        # Common plotting parameters
        plot_kwargs = {"edgecolors": "w", "linewidths": 0.5}

        # Plot clustering results
        plot_clusters(
            X,
            labels=labels,
            use_tsne=False,
            title=title,
            cmap=viz_config["cmap"],
            alpha=viz_config["alpha"],
            point_size=viz_config["point_size"],
            ax=ax,
            plot_kwargs=plot_kwargs,
        )

        # Add dataset name as row title if first column
        if j == 0:
            ax.set_ylabel(f"Dataset: {dataset_name}", fontsize=12)

# Add main title
fig.suptitle("Clustering Algorithm Comparison", fontsize=16)

# Adjust layout
plt.tight_layout()  # Make room for the suptitle

# Save the figure
output_path = f"{output_dir}/quick_shift_example.png"
plt.savefig(output_path, dpi=viz_config["dpi"], bbox_inches="tight")
print(f"\nVisualization saved to {output_path}")

print("\nClustering comparison example completed successfully!")
