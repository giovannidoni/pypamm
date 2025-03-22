# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from pypamm.distance_metrics cimport calculate_distance

# Helper functions for Union-Find
cdef int find_root(int v, int[:] parent) except? -1 nogil:
    """
    Find the root of the set containing v with path compression.
    Returns -1 only in case of error.
    """
    while parent[v] != v:
        parent[v] = parent[parent[v]]  # Path compression
        v = parent[v]
    return v

cdef void union_sets(int v1, int v2, int[:] parent) noexcept nogil:
    """
    Union the sets containing v1 and v2.
    This function cannot raise exceptions.
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
    - X: Data matrix (N x D)
    - metric: Distance metric to use

    Returns:
    - mst_edges: Array of MST edges [(i, j, distance), ...]
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
