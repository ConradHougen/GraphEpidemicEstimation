# File containing general utility functions for manipulating graphs
import numpy as np

# True if matrix is square
def is_matrix_2d_square(A):
    dim = A.shape
    if len(dim) != 2:
        print("Matrix is not 2 dimensional")
        return False
    elif dim[0] != dim[1]:
        print("m = {} but n = {}, so the matrix is not square".format(dim[0], dim[1]))
        return False
    else:
        return True

# True if matrix is symmetric
def is_symmetric_matrix(A):
    return (A.transpose() == A).all()

# True if the adjacency matrix has a zero diagonal (no loops)
def does_not_have_loops(adj_mat):
    return (np.diagonal(adj_mat) == 0).all()

# True if the adjacency matrix represents a valid simple graph
# Confirms edges are unweighted (1 or 0)
# Confirms edges are undirected (symmetric matrix)
# Confirms there are no loops (diagonal is zero)
def is_simple_graph(adj_mat):
    valid = is_matrix_2d_square(adj_mat)
    if not valid:
        print("Matrix is not a valid adjacency matrix for a graph")
        return False

    # Confirm that all values are 0 or 1
    valid_mat = np.isin(adj_mat, {0, 1})
    valid = np.all(valid_mat)
    if not valid:
        print("Matrix may have weighted edges")
        return False

    # Confirm edges are undirected (symmetric matrix)
    valid = is_symmetric_matrix(adj_mat)
    if not valid:
        print("Matrix is not symmetric, so this is a digraph")
        return False

    # Confirm there are no loops (zero diagonal)
    valid = does_not_have_loops(adj_mat)
    if not valid:
        print("Matrix has nonzero values on diagonal, so graph has loops")
        return False

    # All conditions passed, so return True
    return True

# Number of vertices
def get_num_vertices(adj_mat):
    return adj_mat.shape[0]

# Number of edges in an undirected graph with no loops.
def get_num_edges_simple_graph(adj_mat):
    n = adj_mat.shape[0]

    if not is_simple_graph(adj_mat):
        print("This is not a simple graph, use correct function")
        return -1
    else:
        upper_tri_idx = np.triu_indices(n, 1)
        return np.count_nonzero(adj_mat[upper_tri_idx])

# Total sum of possibly weighted edges of a graph
def get_total_graph_edge_weight(adj_mat):
    return np.sum(adj_mat)

