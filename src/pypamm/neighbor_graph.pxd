import numpy as np
cimport numpy as np
from pypamm.distance_metrics cimport dist_func_t

# Declare the public API
cpdef tuple build_knn_graph(np.ndarray[np.float64_t, ndim=2] X, int k, str metric="*", 
                     object inv_cov=None, bint include_self=False, int n_jobs=1)

cpdef compute_knn_for_point(np.ndarray[np.float64_t, ndim=2] X, int i, int k,
                         np.ndarray[np.int32_t, ndim=2] indices,
                         np.ndarray[np.float64_t, ndim=2] distances,
                         str metric, np.ndarray[np.float64_t, ndim=2] inv_cov_arr,
                         bint include_self)

cpdef object build_neighbor_graph(
    np.ndarray[np.float64_t, ndim=2] X,
    int k,
    np.ndarray[np.float64_t, ndim=2] inv_cov = None,
    str metric="*",
    str method="*",
    str graph_type="*"
) 