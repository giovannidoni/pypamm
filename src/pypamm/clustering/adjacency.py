import numpy as np
from typing import Tuple, List, Optional
from numpy.typing import NDArray, ArrayLike
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def merge(
    adj: NDArray[np.float64], 
    ic: NDArray[np.int32], 
    thresh: float, 
    N: int
) -> NDArray[np.int32]:
    """
    Efficiently merges clusters based on adjacency matrix.

    Parameters:
    - adj: (Nc x Nc) Adjacency matrix of cluster relationships.
    - ic: Initial cluster labels.
    - thresh: Threshold for merging clusters.
    - N: Number of samples.

    Returns:
    - Updated cluster labels after merging.
    """
    Nc = len(adj)

    # Step 1: Threshold adjacency matrix to create a graph
    adj_binary = adj > thresh

    # Step 2: Find connected components in the graph
    n_components, labels = connected_components(csr_matrix(adj_binary), directed=False)

    # Step 3: Create a mapping from old cluster IDs to new cluster IDs
    cluster_map = np.zeros(Nc, dtype=np.int32)
    for i in range(Nc):
        cluster_map[i] = labels[i]

    # Step 4: Apply the mapping to the initial cluster labels
    new_labels = np.zeros(N, dtype=np.int32)
    for i in range(N):
        if ic[i] >= 0:  # Only map valid cluster IDs
            new_labels[i] = cluster_map[ic[i]]
        else:
            new_labels[i] = -1  # Keep invalid cluster IDs as -1

    return new_labels

def compute_adjacency(
    prob: NDArray[np.float64], 
    clus: NDArray[np.int32], 
    boot: Optional[NDArray[np.int32]] = None
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute adjacency matrix between clusters based on probability distributions.

    Parameters:
    - prob: (N,) Probability values for each data point.
    - clus: (N,) Cluster assignments for each data point.
    - boot: (N,) Optional bootstrap indices.

    Returns:
    - adj: (Nc x Nc) Adjacency matrix.
    - pks: (Nc,) Probability mass for each cluster.
    """
    # If bootstrap indices are provided, use them to filter the data
    if boot is not None:
        prob = prob[boot]
        clus = clus[boot]

    # Get unique cluster IDs and count
    unique_clusters = np.unique(clus)
    Nc = len(unique_clusters)

    # Initialize adjacency matrix and probability mass array
    adj = np.zeros((Nc, Nc), dtype=np.float64)
    pks = np.zeros(Nc, dtype=np.float64)

    # Compute probability mass for each cluster
    for i, c in enumerate(unique_clusters):
        mask = (clus == c)
        pks[i] = np.sum(prob[mask])

    # Normalize probability masses
    pks = pks / np.sum(pks)

    # Compute adjacency matrix
    for i in range(Nc):
        for j in range(i, Nc):
            if i == j:
                adj[i, j] = 1.0  # Self-adjacency is always 1
            else:
                # Compute overlap between clusters
                ci = unique_clusters[i]
                cj = unique_clusters[j]
                mask_i = (clus == ci)
                mask_j = (clus == cj)
                
                # Compute overlap as minimum of probability masses
                overlap = min(np.sum(prob[mask_i]), np.sum(prob[mask_j]))
                
                # Normalize by total probability
                adj[i, j] = overlap / np.sum(prob)
                adj[j, i] = adj[i, j]  # Symmetric matrix

    return adj, pks