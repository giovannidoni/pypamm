import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def merge(adj: NDArray[np.float64], ic: NDArray[np.int32], thresh: float, N: int) -> NDArray[np.int32]:
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

    # Ensure cluster labels are within bounds
    max_label = np.max(ic)
    if max_label >= Nc:
        # Remap cluster labels to be contiguous and within bounds
        unique_labels = np.unique(ic)
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        remapped_ic = np.array([label_map[label] for label in ic], dtype=np.int32)

        # Create a new adjacency matrix with the correct size
        new_adj = np.zeros((len(unique_labels), len(unique_labels)), dtype=np.float64)
        for i, old_i in enumerate(unique_labels):
            for j, old_j in enumerate(unique_labels):
                if old_i < Nc and old_j < Nc:
                    new_adj[i, j] = adj[old_i, old_j]
                elif i == j:
                    new_adj[i, j] = 1.0  # Self-adjacency is always 1
                else:
                    new_adj[i, j] = 0.0  # No adjacency for new clusters

        # Update variables for the rest of the function
        adj = new_adj
        ic = remapped_ic
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
        if ic[i] >= 0 and ic[i] < Nc:  # Only map valid cluster IDs
            new_labels[i] = cluster_map[ic[i]]
        else:
            # Keep invalid cluster IDs (negative values) as they are
            new_labels[i] = ic[i] if ic[i] < 0 else n_components

    return new_labels


def compute_adjacency(
    prob: NDArray[np.float64], clus: NDArray[np.int32], boot: NDArray[np.int32] | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
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
        mask = clus == c
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
                mask_i = clus == ci
                mask_j = clus == cj

                # Compute overlap as minimum of probability masses
                overlap = min(np.sum(prob[mask_i]), np.sum(prob[mask_j]))

                # Normalize by total probability
                adj[i, j] = overlap / np.sum(prob)
                adj[j, i] = adj[i, j]  # Symmetric matrix

    return adj, pks
