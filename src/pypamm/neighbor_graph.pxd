import numpy as np
cimport numpy as np

# Declare the public API
cpdef tuple build_knn_graph(np.ndarray[np.float64_t, ndim=2] X, int n_neigh, str metric=*, int k=*,
                     object inv_cov=*, bint include_self=*, int n_jobs=*)

cpdef compute_knn_for_point(np.ndarray[np.float64_t, ndim=2] X,
                         int i,
                         int n_neigh,
                         np.ndarray[np.int32_t, ndim=2] indices,
                         np.ndarray[np.float64_t, ndim=2] distances,
                         str metric,
                         int k,
                         np.ndarray[np.float64_t, ndim=2] inv_cov_arr,
                         bint include_self)

cpdef object build_neighbor_graph(
    np.ndarray[np.float64_t, ndim=2] X,
    int n_neigh,
    str method=*,
    str graph_type=*,
    str metric=*,
    int k=*,
    object inv_cov=*
)
