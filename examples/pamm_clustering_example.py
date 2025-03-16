#!/usr/bin/env python
"""
PAMM Clustering Example with Bootstrapping

This example demonstrates how to use the PAMMCluster class for robust clustering
with bootstrapping and adjacency-based merging. PAMM (Probabilistic Analysis of
Molecular Motifs) is particularly useful for:

- Finding robust clusters in noisy data
- Automatically determining the number of clusters
- Handling high-dimensional data common in molecular simulations
- Providing probabilistic assignments for data points

The example uses a YAML configuration file for all parameters.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from data_generator import generate_dataset
from sklearn.cluster import KMeans

from pypamm.clustering.pamm import PAMMCluster

# Load configuration from YAML file
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_path) as f:
    import yaml

    config = yaml.safe_load(f)

# Generate the dataset
X = generate_dataset(config)

# Set up the figure
viz_config = config["visualization"]
fig, axes = plt.subplots(2, 2, figsize=tuple(viz_config["figsize"]))
axes = axes.flatten()

# Get methods to compare (PAMM methods + K-means)
methods = config["pamm_methods"] + [{"name": "K-means (k=4)", "is_kmeans": True}]

# Run each clustering method and plot results
for i, method_config in enumerate(methods):
    method_name = method_config["name"]
    ax = axes[i]

    # Apply clustering
    start_time = time.time()

    if method_config.get("is_kmeans", False):
        # K-means clustering
        kmeans_params = config["kmeans"]
        kmeans = KMeans(**kmeans_params)
        cluster_labels = kmeans.fit_predict(X)
    else:
        # PAMM clustering
        pamm_params = method_config["params"]
        pamm = PAMMCluster(**pamm_params)
        cluster_labels = pamm.fit(X)

    end_time = time.time()
    clustering_time = end_time - start_time

    # Count number of clusters and their sizes
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels)
    cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels]
    avg_cluster_size = np.mean(cluster_sizes)

    # Print statistics
    print(f"\n{method_name}:")
    print(f"  Found {n_clusters} clusters with sizes: {cluster_sizes}")
    print(f"  Average cluster size: {avg_cluster_size:.1f} points")
    print(f"  Clustering time: {clustering_time:.4f} seconds")

    # Plot the results
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=cluster_labels,
        cmap=viz_config["cmap"],
        alpha=viz_config["alpha"],
        s=viz_config["point_size"],
        edgecolors="w",
        linewidths=0.5,
    )

    # Add title with method name and number of clusters
    ax.set_title(f"{method_name}\n({n_clusters} clusters, {clustering_time:.2f}s)")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    # Add legend for cluster labels
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
    ax.add_artist(legend1)

# Adjust layout and save figure
plt.tight_layout()
output_path = viz_config["pamm_output_path"]
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=viz_config["dpi"], bbox_inches="tight")

# Print explanation of the PAMM algorithm with focus on bootstrapping
print("\nExplanation of PAMM Clustering with Bootstrapping:")
print("---------------------------------------------")
print("PAMM (Probabilistic Analysis of Molecular Motifs) is a robust clustering algorithm that:")
print("1. Uses grid selection to reduce the dataset size")
print("2. Estimates density using Kernel Density Estimation (KDE)")
print("3. Applies Quick Shift clustering to find density modes")
print("4. Uses bootstrapping to improve robustness against noise")
print("5. Merges similar clusters based on adjacency")

print("\nHow Bootstrapping Works in PAMM:")
print("1. Multiple clustering runs are performed on resampled data")
print("2. Each run creates a different clustering due to random resampling")
print("3. An adjacency matrix is computed to identify consistent clusters")
print("4. Similar clusters are merged based on the adjacency matrix")
print("5. The result is more robust to noise and outliers")

print("\nKey Parameters:")
print("- n_grid: Number of grid points for density estimation")
print("- k_neighbors: Number of neighbors for graph construction")
print("- bandwidth: Controls the smoothness of the density estimation")
print("- merge_threshold: Threshold for merging similar clusters")
print("- n_bootstrap: Number of bootstrap iterations (more iterations = more robust)")

print("\nExample completed successfully!")
