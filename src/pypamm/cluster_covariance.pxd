# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

import numpy as np
cimport numpy as np
from libc.math cimport sqrt

# Declare the public function that will be exposed to Python
cpdef np.ndarray[np.float64_t, ndim=3] compute_cluster_covariance(
    np.ndarray[np.float64_t, ndim=2] X,
    np.ndarray[np.int32_t, ndim=1] cluster_labels
) 