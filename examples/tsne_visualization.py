#!/usr/bin/env python
"""
t-SNE Visualization Example

This example demonstrates how to generate a high-dimensional dataset
and visualize it using t-SNE dimensionality reduction. It shows how
the t-SNE visualization can reveal cluster structure in high-dimensional
data that would otherwise be difficult to visualize.
"""

import argparse
import os

import matplotlib.pyplot as plt
from data_generator import generate_dataset, list_dataset_sections
from sklearn.cluster import KMeans


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Visualize high-dimensional datasets using t-SNE")
    parser.add_argument(
        "--dataset", type=str, default="dataset_high_dim", help="Dataset section from config.yaml to use"
    )
    parser.add_argument(
        "--output", type=str, default="tsne_visualization.png", help="Output file path for the visualization"
    )
    args = parser.parse_args()

    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        import yaml

        config = yaml.safe_load(f)

    # Check if dataset exists
    if args.dataset not in config:
        print(f"Dataset '{args.dataset}' not found in config.yaml")
        print("Available datasets:")
        for section in list_dataset_sections():
            print(f"  - {section}")
        return

    # Get dataset dimensionality
    n_dimensions = config[args.dataset]["n_dimensions"]
    n_clusters = config[args.dataset]["n_clusters"]

    # Generate dataset
    print(f"Generating {args.dataset} dataset...")
    if n_dimensions > 2:
        X, X_2d = generate_dataset(args.dataset, return_tsne=True)
    else:
        X = generate_dataset(args.dataset)
        X_2d = X

    # Apply K-means clustering for coloring
    print(f"Applying K-means clustering with k={n_clusters}...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # Set up the figure
    plt.figure(figsize=(12, 10))

    # Plot the results
    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1], c=cluster_labels, cmap="viridis", alpha=0.8, s=30, edgecolors="w", linewidths=0.5
    )

    # Add title and labels
    title = f"{args.dataset}\n"
    if n_dimensions > 2:
        title += f"{n_dimensions}D data visualized with t-SNE"
    else:
        title += f"{n_dimensions}D data"

    plt.title(title)
    plt.xlabel("t-SNE 1" if n_dimensions > 2 else "Feature 1")
    plt.ylabel("t-SNE 2" if n_dimensions > 2 else "Feature 2")

    # Add colorbar for cluster labels
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster")

    # Add legend for cluster labels
    legend1 = plt.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
    plt.gca().add_artist(legend1)

    # Add grid
    plt.grid(True, alpha=0.3)

    # Save figure
    plt.tight_layout()
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches="tight")

    print(f"Visualization saved to {args.output}")

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
