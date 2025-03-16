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

The example focuses on how bootstrapping improves clustering robustness,
especially in the presence of noise.
"""

import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

from pypamm.clustering.pamm import PAMMCluster

# Set random seed for reproducibility
np.random.seed(42)


# Generate synthetic data with 4 clusters and noise
def generate_dataset(n_samples=1000, n_features=2, n_clusters=4, noise_fraction=0.3):
    """Generate a synthetic dataset with clusters and noise."""
    # Generate clustered data
    n_clustered = int(n_samples * (1 - noise_fraction))
    X_clustered, y_true = make_blobs(
        n_samples=n_clustered, n_features=n_features, centers=n_clusters, cluster_std=0.5, random_state=42
    )

    # Generate noise data (uniform distribution across the range of clustered data)
    n_noise = n_samples - n_clustered
    min_vals = np.min(X_clustered, axis=0)
    max_vals = np.max(X_clustered, axis=0)
    X_noise = np.random.uniform(low=min_vals, high=max_vals, size=(n_noise, n_features))

    # Combine clustered data and noise
    X = np.vstack([X_clustered, X_noise])

    # Shuffle the data
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X = X[indices]

    print(f"Generated dataset with {n_samples} points in {n_features}D space")
    print(f"Approximately {noise_fraction * 100:.0f}% classified as noise")

    return X


# Generate the dataset
X = generate_dataset(n_samples=1000, n_features=2, n_clusters=4, noise_fraction=0.3)

# Define clustering methods to compare
# Format: (name, use_bootstrap, graph_type, n_bootstrap, merge_threshold)
methods = [
    ("PAMM (No Bootstrap)", False, None, 0, 0.8),
    ("PAMM with KNN Graph", False, "knn", 0, 0.8),
    ("PAMM with 5 Bootstrap Iterations", True, None, 5, 0.6),  # Adjusted merge threshold for bootstrap
    ("K-means (k=4)", False, None, 0, 0.8),
]

# Set up the figure
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

# Run each clustering method and plot results
for i, (method_name, use_bootstrap, graph_type, n_bootstrap, merge_threshold) in enumerate(methods):
    ax = axes[i]

    # Apply clustering
    start_time = time.time()

    if method_name.startswith("K-means"):
        # K-means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
    else:
        # PAMM clustering
        pamm = PAMMCluster(
            n_grid=100,
            k_neighbors=5,
            metric="euclidean",
            merge_threshold=merge_threshold,  # Use the method-specific merge threshold
            bootstrap=use_bootstrap,
            n_bootstrap=n_bootstrap,
            n_jobs=-1,  # Use all available cores
            graph_type=graph_type,
            bandwidth=0.5,  # Specify bandwidth for KDE
        )

        result = pamm.fit(X)

        # Handle different return types
        if isinstance(result, tuple) and len(result) == 2:
            cluster_labels, prob = result
        else:
            cluster_labels = result

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
        X[:, 0], X[:, 1], c=cluster_labels, cmap="viridis", alpha=0.8, s=30, edgecolors="w", linewidths=0.5
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
plt.savefig("pamm_clustering_example.png", dpi=300, bbox_inches="tight")
plt.show()

# Copy the image to the docs/images directory
os.makedirs("docs/images", exist_ok=True)
shutil.copy("pamm_clustering_example.png", "docs/images/")
print("Image copied to docs/images/pamm_clustering_example.png")

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

print("\nAdvantages of Bootstrapping in PAMM:")
print("- Reduces sensitivity to noise and outliers")
print("- Improves stability of clustering results")
print("- Identifies core clusters that persist across resamples")
print("- Provides more reliable results for noisy data")
print("- Automatically determines the appropriate number of clusters")

print("\nExample completed successfully!")
