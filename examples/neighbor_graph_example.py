#!/usr/bin/env python
"""
Neighbor Graph Example

This example demonstrates how to use the neighbor graph functions from pypamm
to build different types of neighborhood graphs for a dataset.

Neighbor graphs are useful for:
- Capturing the structure and connectivity of data
- Preprocessing for clustering algorithms
- Dimensionality reduction
- Manifold learning
- Feature extraction
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import os
import argparse
from pypamm import build_neighbor_graph, build_knn_graph
from data_generator import generate_dataset, load_config

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Neighbor Graph Example')
    parser.add_argument('--config', default='clustered',
                        help='Configuration name or path to YAML file (default: clustered)')
    parser.add_argument('--output', '-o', default='neighbor_graph_example.png',
                        help='Output file path for the visualization')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of neighbors for KNN graph (default: 5)')
    
    args = parser.parse_args()
    
    # Load the configuration and generate dataset
    try:
        if os.path.exists(args.config):
            print(f"Loading configuration from file: {args.config}")
            config = args.config
        else:
            print(f"Loading predefined configuration: {args.config}")
            config = load_config(args.config)
        
        X = generate_dataset(config)
        print(f"Generated dataset with {X.shape[0]} points in {X.shape[1]}D space")
    except Exception as e:
        print(f"Error generating dataset: {e}")
        return
    
    # Define different graph types to try
    graph_types = ["knn", "gabriel"]
    
    # Create a figure for visualization
    plt.figure(figsize=(12, 6))
    
    # Plot each graph type
    for i, graph_type in enumerate(graph_types):
        plt.subplot(1, 2, i+1)
        
        # Build the graph
        if graph_type == "knn":
            # For KNN, use the specialized function
            k = args.k
            indices, distances = build_knn_graph(X, k=k)
            # Convert to sparse matrix
            rows = np.repeat(np.arange(X.shape[0]), k)
            cols = indices.flatten()
            data = np.ones_like(cols, dtype=float)
            graph = sparse.csr_matrix((data, (rows, cols)), shape=(X.shape[0], X.shape[0]))
            title = f"K-Nearest Neighbors (k={k})"
        else:
            # For other graph types, use the general function
            # Note: k parameter is required even for non-KNN graphs
            k = args.k  # This is used for initial neighbor search but doesn't affect the final graph structure
            graph = build_neighbor_graph(X, k=k, graph_type=graph_type)
            title = f"{graph_type.replace('_', ' ').title()} Graph"
        
        # Plot the points
        plt.scatter(X[:, 0], X[:, 1], s=30)
        
        # Plot the edges
        rows, cols = graph.nonzero()
        for r, c in zip(rows, cols):
            plt.plot([X[r, 0], X[c, 0]], [X[r, 1], X[c, 1]], 'k-', alpha=0.2)
        
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        
        # Print some statistics
        n_edges = graph.nnz // 2  # Divide by 2 because the graph is symmetric
        avg_degree = graph.nnz / X.shape[0]
        print(f"{title}:")
        print(f"  - Number of edges: {n_edges}")
        print(f"  - Average node degree: {avg_degree:.2f}")
        print(f"  - Graph density: {graph.nnz / (X.shape[0] * X.shape[0]):.4f}")
    
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Figure saved as '{args.output}'")
    
    # Print explanation of the different graph types
    print("\nExplanation of Graph Types:")
    print("---------------------------")
    print("1. K-Nearest Neighbors (KNN):")
    print("   - Connects each point to its k nearest neighbors")
    print("   - Simple and efficient, but can create disconnected components")
    print("   - Good for local structure, but may miss global structure")
    print("   - Parameter k controls the density of connections")
    
    print("\n2. Gabriel Graph:")
    print("   - Two points are connected if no other point lies in the circle with")
    print("     diameter defined by the two points")
    print("   - Captures both local and global structure")
    print("   - No parameters to tune")
    print("   - Tends to create more connections than RNG but fewer than Delaunay")
    
    print("\nChoosing the Right Graph Type:")
    print("- KNN: When you need control over the number of connections")
    print("- Gabriel: When you want a balance between sparsity and connectivity")
    
    # Example of using the graph for a simple application: connected components
    print("\nExample Application: Finding Connected Components")
    graph = build_neighbor_graph(X, k=args.k, graph_type="gabriel")
    n_components, labels = sparse.csgraph.connected_components(graph, directed=False)
    print(f"The Gabriel graph has {n_components} connected components")
    
    # Count points in each component
    for i in range(n_components):
        print(f"Component {i+1} has {np.sum(labels == i)} points")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main() 