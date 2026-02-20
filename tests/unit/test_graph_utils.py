import numpy as np

from asunder.utils.graph import (
    contract_adj_matrix_new,
    expand_z_matrix,
    partition_matrix_to_vector,
    partition_vector_to_2d_matrix,
)


def test_partition_vector_roundtrip():
    labels = np.array([0, 0, 1, 2, 2])
    z = partition_vector_to_2d_matrix(labels)
    got = partition_matrix_to_vector(z)
    assert got.shape == labels.shape
    assert np.all(z == partition_vector_to_2d_matrix(got))


def test_contract_and_expand_shape():
    A = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=float,
    )
    A_sup, node2comp = contract_adj_matrix_new(A, must_link=[(0, 1)])
    assert A_sup.shape[0] <= A.shape[0]
    z_small = np.eye(A_sup.shape[0], dtype=int)
    z_full = expand_z_matrix(z_small, node2comp)
    assert z_full.shape == A.shape

