import numpy as np
cimport numpy as np

# Declare the public API
cpdef tuple build_knn_graph(np.ndarray[np.float64_t, ndim=2] X, int k, str metric=*,
                     object inv_cov=*, bint include_self=*, int n_jobs=*)

cpdef compute_knn_for_point(np.ndarray[np.float64_t, ndim=2] X, int i, int k,
                         np.ndarray[np.int32_t, ndim=2] indices,
                         np.ndarray[np.float64_t, ndim=2] distances,
                         str metric, np.ndarray[np.float64_t, ndim=2] inv_cov_arr,
                         bint include_self)

cpdef object build_neighbor_graph(
    np.ndarray[np.float64_t, ndim=2] X,
    int k,
    object inv_cov=*,
    str metric=*,
    str method=*,
    str graph_type=*
)
