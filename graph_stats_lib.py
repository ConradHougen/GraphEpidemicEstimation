# Library of helper functions related to computing graph statistics

import scipy.linalg as la


# K largest eigenvalues of an adjacency matrix
def compute_k_largest_eigenvalues(adj_mat, k):
    print("Computing {} largest eigenvalues".format(k))
    n = adj_mat.shape[0]
    if k < n:
        largest_k_eigvals = la.eigvalsh(adj_mat, eigvals=(n - k, n - 1))
    else:
        largest_k_eigvals = la.eigvalsh(adj_mat)

    # Sort the eigenvalues from largest to smallest
    idx = largest_k_eigvals.argsort()[::-1]
    largest_k_eigvals = largest_k_eigvals[idx]
    return largest_k_eigvals


# Degree distribution
def compute_degree_distribution(adj_mat):
    print("Computing degree distribution")

# Clustering coefficient
def compute_global_clustering_coefficient(adj_mat):
    print("Computing global clustering coefficient")

