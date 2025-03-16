#!/usr/bin/env python
"""
QuickShift Clustering Example

This example demonstrates how to use the QuickShift clustering algorithm
for finding clusters in a dataset. QuickShift is a mode-seeking algorithm
that associates each data point with a local mode of the density.

The example uses a YAML configuration file for all parameters.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from data_generator import generate_dataset

from pypamm.clustering.quickshift import quick_shift

# Load configuration from YAML file
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_path) as f:
    import yaml

    config = yaml.safe_load(f)

# Generate the dataset
X = generate_dataset(config)

# Get QuickShift parameters from config
qs_config = config["quickshift"]
bandwidth = qs_config["bandwidth"]
k_neighbors = qs_config["k_neighbors"]

# Apply QuickShift clustering
start_time = time.time()
cluster_labels, probabilities = quick_shift(X, bandwidth=bandwidth, k_neighbors=k_neighbors)
end_time = time.time()
clustering_time = end_time - start_time

# Count number of clusters and their sizes
unique_labels = np.unique(cluster_labels)
n_clusters = len(unique_labels)
cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels]
avg_cluster_size = np.mean(cluster_sizes)

# Print statistics
print("\nQuickShift Clustering:")
print(f"  Found {n_clusters} clusters with sizes: {cluster_sizes}")
print(f"  Average cluster size: {avg_cluster_size:.1f} points")
print(f"  Clustering time: {clustering_time:.4f} seconds")

# Set up the figure
viz_config = config["visualization"]
fig, ax = plt.subplots(figsize=tuple(viz_config["figsize"]))

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
ax.set_title(f"QuickShift Clustering\n({n_clusters} clusters, {clustering_time:.2f}s)")
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")

# Add legend for cluster labels
legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
ax.add_artist(legend1)

# Adjust layout and save figure
plt.tight_layout()
output_path = viz_config["quickshift_output_path"]
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=viz_config["dpi"], bbox_inches="tight")

# Print explanation of the QuickShift algorithm
print("\nExplanation of QuickShift Clustering:")
print("---------------------------------------------")
print("QuickShift is a mode-seeking clustering algorithm that:")
print("1. Estimates density using Kernel Density Estimation (KDE)")
print("2. Associates each point with its nearest neighbor of higher density")
print("3. Forms clusters by connecting points that share the same mode")
print("4. Automatically determines the number of clusters based on density modes")

print("\nKey Parameters:")
print(f"- bandwidth: {bandwidth} (Controls the smoothness of the density estimation)")
print(f"- k_neighbors: {k_neighbors} (Number of neighbors for density estimation)")

print("\nExample completed successfully!")
