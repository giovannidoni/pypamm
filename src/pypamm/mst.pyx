# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

"""
Cython implementation of minimum spanning tree (MST) algorithms.
This module provides efficient computation of MST using Kruskal's algorithm.
"""

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from pypamm.lib.distance cimport calculate_distance
from libc.math cimport sqrt

# Helper functions for Union-Find
cdef int find_root(int v, int[:] parent) except? -1 nogil:
    """
    Find the root of the set containing v with path compression.

    Parameters:
    - v: Vertex to find the root for
    - parent: Array representing the parent of each vertex in the disjoint set

    Returns:
    - Root of the set containing v, or -1 in case of error
    """
    while parent[v] != v:
        parent[v] = parent[parent[v]]  # Path compression
        v = parent[v]
    return v

cdef void union_sets(int v1, int v2, int[:] parent) noexcept nogil:
    """
    Union the sets containing v1 and v2.

    Parameters:
    - v1: First vertex
    - v2: Second vertex
    - parent: Array representing the parent of each vertex in the disjoint set

    Notes:
    - This function cannot raise exceptions.
    """
    cdef int root1 = find_root(v1, parent)
    cdef int root2 = find_root(v2, parent)
    if root1 != root2:
        parent[root1] = root2

# ------------------------------------------------------------------------------
# 1. Minimum Spanning Tree (MST) Using Kruskal's Algorithm
# ------------------------------------------------------------------------------
cpdef np.ndarray[np.float64_t, ndim=2] build_mst(np.ndarray[np.float64_t, ndim=2] X, str metric="euclidean"):
    """
    Builds the Minimum Spanning Tree (MST) for the dataset using Kruskal's Algorithm.

    Parameters:
    - X: Data matrix (N x D) of points to connect with MST
    - metric: Distance metric to use ("euclidean", "manhattan", "chebyshev", etc.)

    Returns:
    - mst_edges: Array of MST edges (N-1 x 3) where each row contains [source_idx, target_idx, distance]

    Notes:
    - The implementation uses Kruskal's algorithm with Union-Find data structure
    - Time complexity is O(N² log N) where N is the number of points
    """
    cdef Py_ssize_t N = X.shape[0]
    cdef Py_ssize_t i, j
    cdef list edges = []
    cdef np.ndarray[np.float64_t, ndim=2] distances = np.zeros((N, N), dtype=np.float64)
    cdef np.ndarray[np.int32_t, ndim=1] parent = np.arange(N, dtype=np.int32)
    cdef int[:] parent_view = parent

    # Compute pairwise distances
    for i in range(N):
        for j in range(i + 1, N):
            distances[i, j] = calculate_distance(metric, X[i], X[j])
            distances[j, i] = distances[i, j]
            edges.append((distances[i, j], i, j))

    # Sort edges by weight
    edges.sort()

    # Build MST using Kruskal's Algorithm
    cdef list mst_edges = []
    cdef double weight
    cdef int u, v

    for edge in edges:
        weight, u, v = edge
        if find_root(u, parent_view) != find_root(v, parent_view):
            union_sets(u, v, parent_view)
            mst_edges.append((u, v, weight))

    # Convert to numpy array
    cdef np.ndarray[np.float64_t, ndim=2] mst_edges_array = np.array(mst_edges, dtype=np.float64)
    return mst_edges_array
