"""
PyPAMM: Python Package for Advanced Manifold Methods

A library for manifold learning, clustering, and density estimation.
"""

__version__ = "0.1.0"

# Define the modules we want to import
__all__ = [
    "select_grid_points",
    "build_neighbor_graph",
    "build_knn_graph",
    "get_distance_function",
    "compute_cluster_covariance",
    "merge_clusters",
    "reindex_clusters",
    "quick_shift",
    "quick_shift_kde",
    "build_mst",
    # Density module
    "gauss_prepare",
    "compute_kde",
    "kde_cutoff",
    "kde_bootstrap_error",
    "kde_output",
]

# Import the modules
from pypamm.clustering.cluster_utils_wrapper import compute_cluster_covariance, merge_clusters, reindex_clusters

# Import density module
from pypamm.density import compute_kde, gauss_prepare, kde_bootstrap_error, kde_cutoff, kde_output
from pypamm.grid_selection import select_grid_points
from pypamm.lib.distance_wrapper import get_distance_function
from pypamm.mst_wrapper import build_mst
from pypamm.neighbor_graph import build_neighbor_graph
from pypamm.neighbor_graph_wrapper import build_knn_graph
from pypamm.quick_shift_wrapper import quick_shift, quick_shift_kde
