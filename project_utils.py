# File with helper functions for managing input and output (reading and writing files, plotting, etc)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from graph_stats_lib import compute_k_largest_eigenvalues

def create_node_df_from_metadata_file(metadata_filename):
    with open(metadata_filename, 'r') as f:
        df = pd.DataFrame(l.rstrip().split() for l in f)
        df = pd.to_numeric(df[0])
        df = df.sort_values()
        df = df.reset_index(drop=True)
        return df


def create_tij_df_from_data_file(data_filename):
    with open(data_filename, 'r') as f:
        df = pd.DataFrame(l.rstrip().split() for l in f)
        df = df.apply(pd.to_numeric)
        return df


def map_adj_mat_idx_to_node_id(nodes_df, adj_mat_idx):
    if adj_mat_idx in range(0, len(nodes_df)):
        return nodes_df[adj_mat_idx]
    else:
        print("Error, index {} does not map to a valid node".format(adj_mat_idx))
        return -1


def map_node_id_to_adj_mat_idx(nodes_df, node_id):
    if node_id in set(nodes_df):
        return nodes_df[nodes_df == node_id].index.item()
    else:
        print("Error, invalid node id {}".format(node_id))
        return -1


def create_temporally_aggregated_adj_mat(tij_df, nodes_df):
    n = len(nodes_df)
    A = np.zeros((n, n))

    i_col_idx = 1
    j_col_idx = 2

    num_rows = tij_df.shape[0]

    for row_idx in range(0, num_rows):
        # Pandas dataframe takes column index first, row index second
        i = map_node_id_to_adj_mat_idx(nodes_df, tij_df[i_col_idx][row_idx])
        j = map_node_id_to_adj_mat_idx(nodes_df, tij_df[j_col_idx][row_idx])

        # Increment edge weight
        A[i][j] += 1
        A[j][i] += 1

    return A

def plot_k_largest_eigvals_of_two_matrices(A, B, k):
    largest_k_eigvals_A = compute_k_largest_eigenvalues(A, k)
    largest_k_eigvals_B = compute_k_largest_eigenvalues(B, k)

    plt.figure()
    actual_k = len(largest_k_eigvals_A)
    eig_idx = range(0, actual_k)
    plt.subplot(1, 2, 1)
    plt.plot(eig_idx, largest_k_eigvals_A, 'ro')
    actual_k = len(largest_k_eigvals_A)
    eig_idx = range(0, actual_k)
    plt.subplot(1, 2, 1)
    plt.plot(eig_idx, largest_k_eigvals_B, 'bo')
    plt.show()