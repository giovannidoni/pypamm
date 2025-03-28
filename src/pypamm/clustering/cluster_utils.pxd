# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from cython.parallel cimport prange
from pypamm.lib.distance cimport dist_mahalanobis

# Declare the C-level functions that will be exposed to other Cython modules
cdef double logsumexp(np.ndarray[np.float64_t, ndim=1] arr)

cdef int _find_nearest_cluster(
    int i,
    np.ndarray[np.float64_t, ndim=2] X,
    np.ndarray[np.int32_t, ndim=1] cluster_labels,
    np.ndarray[np.float64_t, ndim=3] cluster_covariances,
    np.ndarray[np.uint8_t, ndim=1] mergeornot
)

# Declare the public functions that will be exposed to Python
cpdef np.ndarray[np.float64_t, ndim=3] compute_cluster_covariance(
    np.ndarray[np.float64_t, ndim=2] X,
    np.ndarray[np.int32_t, ndim=1] cluster_labels
)

cpdef np.ndarray[np.int32_t, ndim=1] merge_clusters(
    np.ndarray[np.float64_t, ndim=2] X,
    np.ndarray[np.float64_t, ndim=1] prob,
    np.ndarray[np.int32_t, ndim=1] cluster_labels,
    np.ndarray[np.float64_t, ndim=3] cluster_covariances,
    double threshold=*
)

cpdef np.ndarray[np.int32_t, ndim=1] reindex_clusters(
    np.ndarray[np.int32_t, ndim=1] cluster_labels
)
