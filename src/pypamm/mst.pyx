# cython: language_level=3, boundscheck=False, wraparound=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from pypamm.distance_metrics cimport (
    dist_func_t, _get_distance_function
)

# Helper functions for Union-Find
cdef int find_root(int v, int[:] parent) nogil:
    while parent[v] != v:
        parent[v] = parent[parent[v]]  # Path compression
        v = parent[v]
    return v

cdef void union_sets(int v1, int v2, int[:] parent) nogil:
    parent[find_root(v1, parent)] = find_root(v2, parent)

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

    # Get the distance function
    cdef dist_func_t dist_func = _get_distance_function(metric)

    # Compute pairwise distances
    for i in range(N):
        for j in range(i + 1, N):
            distances[i, j] = dist_func(X[i], X[j], np.zeros((1,1)))
            distances[j, i] = distances[i, j]
            edges.append((distances[i, j], i, j))

    # Sort edges by distance (needed for Kruskal's Algorithm)
    edges.sort()

    # Construct MST using Kruskal's Algorithm
    cdef list mst_edges = []
    cdef double dist
    for dist, i, j in edges:
        if find_root(i, parent_view) != find_root(j, parent_view):
            union_sets(i, j, parent_view)
            mst_edges.append((i, j, dist))
            if len(mst_edges) == N - 1:
                break  # MST complete

    return np.array(mst_edges, dtype=np.float64)