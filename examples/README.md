# PyPAMM Examples

This directory contains example scripts demonstrating how to use the various algorithms and functions in the PyPAMM (Python Probabilistic Analysis of Molecular Motifs) package.

## Running the Examples

To run these examples, make sure you have PyPAMM installed:

```bash
# From the root directory of the repository
poetry install
```

Then you can run any example using Poetry:

```bash
poetry run python examples/grid_selection_example.py
```

## Available Examples

### Grid Selection

**File:** `grid_selection_example.py`

Demonstrates how to use the `select_grid_points` function to select a subset of points from a dataset based on a grid. This is useful for reducing the number of points in large datasets while preserving the overall structure.

### Neighbor Graph

**File:** `neighbor_graph_example.py`

Shows how to build different types of neighborhood graphs (KNN, Gabriel, Relative Neighborhood, Delaunay) using the `build_neighbor_graph` and `build_knn_graph` functions. These graphs capture the connectivity structure of the data.

### Quick Shift Clustering

**File:** `quick_shift_example.py`

Illustrates the use of the `quick_shift` and `quick_shift_clustering` functions for mode-seeking clustering. Quick Shift automatically determines the number of clusters and can find clusters of arbitrary shape.

### Minimum Spanning Tree (MST)

**File:** `mst_example.py`

Demonstrates how to build and use Minimum Spanning Trees with the `build_mst` function. MSTs provide a sparse representation of the data and can be used for clustering, dimensionality reduction, and outlier detection.

**File:** `test_mst_simple.py`

A simple test script for the MST module that doesn't use matplotlib for visualization. It demonstrates the basic usage of the `build_mst` function and verifies the properties of the resulting MST.

### Complete Pipeline

**File:** `pipeline_example.py`

Shows how to combine multiple PyPAMM algorithms in a complete data analysis pipeline:
1. Grid selection for data reduction
2. Building a neighbor graph on the grid points
3. Constructing a minimum spanning tree
4. Using Quick Shift for clustering

This example demonstrates how the different algorithms can work together to analyze complex datasets efficiently.

## Example Outputs

Each example generates visualizations that are saved as PNG files in the current directory. The examples also print explanations and statistics to the console to help understand the algorithms and their parameters.

## Dependencies

These examples require the following dependencies (automatically installed with PyPAMM):

- NumPy
- SciPy
- Matplotlib

## Additional Resources

For more information about the PyPAMM package and its algorithms, please refer to the main documentation in the repository's README file. 