#!/usr/bin/env python
"""
Quick-Shift Clustering Example

This example demonstrates how to use the Quick-Shift clustering algorithm
from the PyPAMM library and compares it with hierarchical clustering.
Quick-Shift is a mode-seeking algorithm that finds clusters by shifting
points towards higher density regions.

The example uses datasets defined in the example_config.yaml file and visualizes
the clustering results side by side for comparison.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import yaml
from data_generator import generate_dataset
from sklearn.cluster import AgglomerativeClustering

from pypamm.density.kde import compute_kde
from pypamm.quick_shift import quick_shift_clustering

# Load configuration from YAML file
config_path = os.path.join(os.path.dirname(__file__), "example_config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

# Get visualization settings
viz_config = config["visualization"]
output_dir = viz_config["output_dir"]
os.makedirs(output_dir, exist_ok=True)

# Get example-specific configuration
example_config = config["examples"]["quick_shift_example"]
datasets = example_config["datasets"]

# Get algorithm parameters
kde_config = config["algorithms"]["kde"]
qs_config = config["algorithms"]["quick_shift"]
hc_config = config["algorithms"]["hierarchical"]

# Process each dataset
for dataset_name in datasets:
    print(f"\n{'=' * 80}")
    print(f"Processing dataset: {dataset_name}")
    print(f"{'=' * 80}")

    # Generate dataset
    dataset_config = config["datasets"][dataset_name]
    X = generate_dataset(dataset_name)

    # 1. Hierarchical Clustering
    print("\nApplying Hierarchical Clustering...")
    # Get number of clusters from the dataset config
    n_clusters = dataset_config["n_clusters"]
    linkage = hc_config["linkage"]

    # Apply hierarchical clustering
    start_time = time.time()
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    hierarchical_labels = hierarchical.fit_predict(X)
    hierarchical_time = time.time() - start_time
    print(f"Hierarchical clustering completed in {hierarchical_time:.4f} seconds")

    # 2. Quick-Shift clustering
    print("\nApplying Quick-Shift clustering...")
    # Compute KDE for density estimation
    bandwidth = kde_config["bandwidth"]
    start_time = time.time()
    density = compute_kde(X, X, bandwidth)
    kde_time = time.time() - start_time
    print(f"KDE computation time: {kde_time:.4f} seconds")

    # Apply Quick-Shift clustering
    lambda_qs = qs_config["lambda_qs"]
    max_dist = qs_config["max_dist"]
    ngrid = qs_config["ngrid"]
    metric = qs_config["metric"]

    start_time = time.time()
    qs_labels, qs_centers = quick_shift_clustering(
        X, density, ngrid=ngrid, metric=metric, lambda_qs=lambda_qs, max_dist=max_dist
    )
    qs_time = time.time() - start_time
    n_qs_clusters = len(np.unique(qs_labels))
    print(f"Quick-Shift found {n_qs_clusters} clusters in {qs_time:.4f} seconds")

    # Create a side-by-side comparison figure
    figsize = tuple(viz_config["figsize"])
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot hierarchical clustering
    scatter1 = axes[0].scatter(
        X[:, 0],
        X[:, 1],
        c=hierarchical_labels,
        cmap=viz_config["cmap"],
        alpha=viz_config["alpha"],
        s=viz_config["point_size"],
        edgecolors="w",
        linewidths=0.5,
    )
    axes[0].set_title(f"Hierarchical Clustering\n{n_clusters} clusters, linkage={linkage}\n{hierarchical_time:.4f}s")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    axes[0].grid(True, alpha=0.3)

    # Plot Quick-Shift clustering
    scatter2 = axes[1].scatter(
        X[:, 0],
        X[:, 1],
        c=qs_labels,
        cmap=viz_config["cmap"],
        alpha=viz_config["alpha"],
        s=viz_config["point_size"],
        edgecolors="w",
        linewidths=0.5,
    )

    axes[1].set_title(
        f"Quick-Shift Clustering\n{n_qs_clusters} clusters, λ={lambda_qs}, max_dist={max_dist}\n{qs_time:.4f}s"
    )
    axes[1].set_xlabel("Feature 1")
    axes[1].set_ylabel("Feature 2")
    axes[1].grid(True, alpha=0.3)

    # Add overall title
    plt.suptitle(f"Clustering Comparison on {dataset_name}\n{dataset_config['description']}", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Save figure
    output_path = f"{output_dir}/comparison_{dataset_name}.png"
    plt.savefig(output_path, dpi=viz_config["dpi"], bbox_inches="tight")
    print(f"Comparison figure saved to {output_path}")
    plt.close(fig)

    # Print comparison summary
    print("\nClustering Comparison Summary:")
    print("=" * 50)
    print(f"Dataset: {dataset_name}")
    print(f"Description: {dataset_config['description']}")
    print(f"True number of clusters: {dataset_config['n_clusters']}")
    print("-" * 50)
    print(f"Hierarchical: {n_clusters} clusters in {hierarchical_time:.4f}s (linkage={linkage})")
    print(f"Quick-Shift: {n_qs_clusters} clusters in {qs_time:.4f}s (+ {kde_time:.4f}s for KDE)")
    print("=" * 50)

print("\nQuick-Shift example completed successfully!")
print(f"All visualizations saved to {output_dir}/")
