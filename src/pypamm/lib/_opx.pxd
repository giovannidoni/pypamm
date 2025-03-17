# cython: language_level=3
# Declarations for optimized operations in clustering module

cimport numpy as cnp
import numpy as np

# Define common types
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.int64_t ITYPE_t

# Declare the functions from _opx.pyx
cpdef object invmatrix(int D, double[:, ::1] M)
cpdef double trmatrix(int D, double[:, ::1] M)
cpdef double detmatrix(int D, double[:, ::1] M)
cpdef double logdet(int D, double[:, ::1] M)
cpdef double variance(int nsamples, int D, double[:, ::1] x, double[::1] weights)
cpdef object eigval(int D, double[:, ::1] AB)
cpdef double maxeigval(int D, double[:, ::1] AB)
cpdef int factorial(int n)
cpdef object pammrij(int D, double[::1] period, double[::1] ri, double[::1] rj)
cpdef double mahalanobis(int D, double[::1] period, double[::1] x, double[::1] y, double[:, ::1] Qinv)
cpdef double effdim(int D, double[:, ::1] Q)
cpdef object oracle(int D, double N, double[:, ::1] Q)

# Additional utility functions that could be useful for clustering operations

# Function to compute pairwise distances between points
cpdef object compute_pairwise_distances(double[:, ::1] X, str metric="euclidean", double[:, ::1] Qinv=None, double[::1] period=None)

# Function to compute cluster centroids
cpdef object compute_centroids(double[:, ::1] X, long[::1] labels, int n_clusters)

# Function to compute cluster covariance matrices
cpdef object compute_cluster_covariances(double[:, ::1] X, long[::1] labels, int n_clusters, bint shrinkage=True)

# Function to compute silhouette scores
cpdef double compute_silhouette(double[:, ::1] X, long[::1] labels, int n_clusters)

# Function to compute Davies-Bouldin index
cpdef double compute_davies_bouldin(double[:, ::1] X, long[::1] labels, int n_clusters)

# Function to compute Calinski-Harabasz index
cpdef double compute_calinski_harabasz(double[:, ::1] X, long[::1] labels, int n_clusters)
