#!/usr/bin/env python
"""
Data Generator for PyPAMM Examples

This module provides a focused tool to generate synthetic datasets for
clustering algorithms. It reads parameters from a config.yaml file and
generates datasets with specified number of clusters, shape, dimensionality,
and noise level, all normalized to the [0,1] range.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.manifold import TSNE


def load_config(config_name: str = "data_config"):
    """
    Load the configuration file for the data generator.
    """
    config_path = os.path.join(os.path.dirname(__file__), f"{config_name}.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def generate_dataset(config_section: str, config_name: str = "data_config") -> tuple:
    """
    Generate a synthetic dataset for clustering based on parameters from a config section.

    Parameters:
    -----------
    config_section : str
        Identifier for the section in the config file containing dataset parameters.
        The section should include:
        - n_clusters: int, number of clusters to generate
        - n_dimensions: int, dimensionality of the data
        - cluster_std: float, standard deviation of clusters (shape parameter)
        - cluster_uniformity: float, controls how uniform the clusters are (0.0 to 1.0)
          where 1.0 means perfectly spherical clusters and lower values create more elongated clusters
        - population_uniformity: float, controls how uniform the cluster populations are (0.0 to 1.0)
          where 1.0 means all clusters have equal size and 0.0 means one large cluster and the rest are small
        - noise_level: float, fraction of points that are noise (0.0 to 1.0)
        - n_samples: int, total number of data points to generate
    config_name : str, default="data_config"
        Name of the configuration file to use (without .yaml extension)

    Returns:
    --------
    X : np.ndarray
        Generated dataset with shape (n_samples, n_dimensions), normalized to [0,1] range
    """
    # Load configuration from YAML file
    config = load_config(config_name)

    # Get dataset parameters from the specified section
    if config_section not in config:
        raise ValueError(f"Section '{config_section}' not found in {config_name}.yaml")

    dataset_config = config[config_section]

    # Extract parameters with defaults
    n_clusters = dataset_config.get("n_clusters", 4)
    n_dimensions = dataset_config.get("n_dimensions", 2)
    cluster_std = dataset_config.get("cluster_std", 0.1)
    cluster_uniformity = dataset_config.get("cluster_uniformity", 1.0)
    population_uniformity = dataset_config.get("population_uniformity", 1.0)
    noise_level = dataset_config.get("noise_level", 0.1)
    n_samples = dataset_config.get("n_samples", 1000)
    random_seed = dataset_config.get("random_seed", 42)

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    # Calculate number of clustered points and noise points
    n_clustered = int(n_samples * (1 - noise_level))
    n_noise = n_samples - n_clustered

    # Calculate points per cluster based on population_uniformity
    if population_uniformity >= 1.0:
        # Perfectly uniform cluster sizes
        samples_per_cluster = [
            n_clustered // n_clusters + (1 if i < n_clustered % n_clusters else 0) for i in range(n_clusters)
        ]
    else:
        # Non-uniform cluster sizes
        # Create a power distribution where lower uniformity means more skewed sizes
        # At uniformity=0, one cluster gets most points, others get minimum
        min_points_per_cluster = max(5, int(n_clustered * 0.01))  # Ensure at least a few points per cluster
        remaining_points = n_clustered - min_points_per_cluster * n_clusters

        if population_uniformity <= 0.0:
            # One dominant cluster, others minimal
            samples_per_cluster = [min_points_per_cluster] * n_clusters
            samples_per_cluster[0] += remaining_points
        else:
            # Create a power law distribution
            # Higher exponent = more uniform (closer to 1.0)
            exponent = 1.0 / (1.0 - population_uniformity) if population_uniformity < 1.0 else 1.0
            weights = np.array([(1.0 / (i + 1)) ** exponent for i in range(n_clusters)])
            weights = weights / weights.sum()

            # Distribute remaining points according to weights
            extra_points = np.round(weights * remaining_points).astype(int)
            # Adjust for rounding errors
            if extra_points.sum() != remaining_points:
                diff = remaining_points - extra_points.sum()
                extra_points[0] += diff

            samples_per_cluster = [min_points_per_cluster + extra for extra in extra_points]

    # Generate cluster centers
    centers = np.random.rand(n_clusters, n_dimensions)

    # Generate clustered data with varying uniformity
    X_clustered = []
    labels = []

    for i in range(n_clusters):
        # Get number of points for this cluster
        n_points = samples_per_cluster[i]

        # Create a random covariance matrix based on uniformity
        if cluster_uniformity >= 1.0:
            # Perfectly spherical clusters
            cov = np.eye(n_dimensions) * (cluster_std**2)
        else:
            # Create a random covariance matrix with controlled elongation
            # Start with a diagonal matrix
            eigenvalues = np.ones(n_dimensions)

            # Modify eigenvalues to create elongation
            # The lower the uniformity, the more variation in eigenvalues
            if n_dimensions > 1:
                variation = (1.0 - cluster_uniformity) * 5.0  # Scale factor for variation
                eigenvalues = np.random.uniform(1.0 - variation, 1.0 + variation, n_dimensions)
                eigenvalues = np.abs(eigenvalues)  # Ensure positive eigenvalues

            # Create a random rotation matrix
            Q = np.random.randn(n_dimensions, n_dimensions)
            Q, _ = np.linalg.qr(Q)  # Orthogonalize

            # Create covariance matrix: Q * diag(eigenvalues) * Q^T
            cov = Q @ np.diag(eigenvalues) @ Q.T
            cov = cov * (cluster_std**2)  # Scale by cluster_std

        # Generate points from multivariate normal distribution
        cluster_points = np.random.multivariate_normal(mean=centers[i], cov=cov, size=n_points)

        X_clustered.append(cluster_points)
        labels.extend([i] * n_points)

    X_clustered = np.vstack(X_clustered)

    # Generate noise data (uniform distribution)
    X_noise = np.random.uniform(low=0, high=1, size=(n_noise, n_dimensions))

    # Combine clustered data and noise
    X = np.vstack([X_clustered, X_noise])

    # Shuffle the data
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X = X[indices]

    # Print dataset information
    print(f"Generated dataset from '{config_section}' section:")
    print(f"  {n_samples} points in {n_dimensions}D space")
    print(f"  {n_clusters} clusters with std={cluster_std}, uniformity={cluster_uniformity}")

    # Print cluster population information
    if population_uniformity < 1.0:
        print(f"  Non-uniform cluster populations (uniformity={population_uniformity}):")
        for i, size in enumerate(samples_per_cluster):
            print(f"    Cluster {i}: {size} points ({size / n_clustered * 100:.1f}%)")
    else:
        print(f"  Uniform cluster populations: ~{n_clustered // n_clusters} points per cluster")

    print(f"  {noise_level * 100:.1f}% noise")

    return X


def list_dataset_sections():
    """
    List all available dataset sections in the config.yaml file.

    Returns:
    --------
    sections : list
        List of section names that can be used for dataset generation
    """
    config = load_config()

    # Filter sections that have dataset parameters
    for section, params in config.items():
        if isinstance(params, dict) and "n_clusters" in params:
            yield section, params


def plot_clusters(
    X,
    labels=None,
    title=None,
    output_path=None,
    figsize=(12, 10),
    cmap="viridis",
    alpha=0.8,
    point_size=30,
    random_seed=42,
    ax=None,
    plot_kwargs=None,
):
    """
    Plot clusters with optional t-SNE dimensionality reduction.

    Parameters:
    -----------
    X : np.ndarray
        Dataset to visualize, can be any dimensionality
    labels : np.ndarray, optional
        Cluster labels for coloring points
    title : str, optional
        Title for the plot
    output_path : str, optional
        Path to save the figure
    figsize : tuple, default=(12, 10)
        Figure size
    cmap : str, default="viridis"
        Colormap for cluster labels
    alpha : float, default=0.8
        Transparency of points
    point_size : int, default=30
        Size of points
    random_seed : int, default=42
        Random seed for reproducibility
    ax : matplotlib.axes.Axes, optional
        Existing axes to plot on. If None, a new figure and axes will be created.
    plot_kwargs : dict, optional
        Additional keyword arguments to pass to the scatter function

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Check dimensionality
    n_dimensions = X.shape[1]

    # Apply t-SNE if needed
    if n_dimensions > 2:
        print(f"Applying t-SNE to reduce {n_dimensions}D data to 2D for visualization...")

        # Then apply t-SNE for final 2D visualization
        tsne = TSNE(n_components=2, random_state=random_seed)
        X_2d = tsne.fit_transform(X)

    elif n_dimensions == 2:
        # Already 2D, no need for reduction
        X_2d = X

    # Create figure if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Set default scatter parameters
    scatter_kwargs = {"alpha": alpha, "s": point_size, "edgecolors": "w", "linewidths": 0.5}

    # Update with user-provided kwargs if any
    if plot_kwargs is not None:
        scatter_kwargs.update(plot_kwargs)

    # Plot the data
    if labels is not None:
        scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap=cmap, **scatter_kwargs)

        # Add legend for cluster labels
        legend = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
        ax.add_artist(legend)

    else:
        ax.scatter(X_2d[:, 0], X_2d[:, 1], **scatter_kwargs)

    # Set title and labels
    if title:
        ax.set_title(title)

    if n_dimensions > 2:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
    else:
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")

    # Add grid
    ax.grid(True, alpha=0.3)

    # Adjust layout if we created the figure
    if ax is None:
        plt.tight_layout()

    # Save figure if output path provided
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {output_path}")

    return fig


# Example usage
if __name__ == "__main__":
    # Create output directory
    output_dir = "docs/images"
    os.makedirs(output_dir, exist_ok=True)

    for section, config in list_dataset_sections():
        # Generate dataset
        X = generate_dataset(section)
        dimension = config["n_dimensions"]

        # Create a more descriptive title with configuration details
        title_parts = []
        title_parts.append(f"{section}")
        title_parts.append(f"{dimension}D data")

        if "cluster_uniformity" in config:
            cu = config["cluster_uniformity"]
            title_parts.append(f"shape_unif={cu:.1f}")

        if "population_uniformity" in config:
            pu = config["population_uniformity"]
            title_parts.append(f"pop_unif={pu:.1f}")

        title = " - ".join(title_parts)

        # Create filename with configuration key
        filename = f"{section}"
        output_path = f"{output_dir}/{filename}.png"

        # Visualize dataset using plot_clusters
        fig = plot_clusters(
            X,
            labels=None,  # No clustering labels
            title=title,
            output_path=output_path,
        )

        # Close the figure to avoid memory issues
        plt.close(fig)
