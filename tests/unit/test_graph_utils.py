import numpy as np
import pytest

from asunder.base.column_generation.master import compute_f_star
from asunder.base.utils.graph import (
    contract_adj_matrix_new,
    contract_node_pairs,
    contract_partition_matrix,
    expand_z_matrix,
    partition_matrix_to_vector,
    partition_vector_to_2d_matrix,
)


def test_partition_vector_roundtrip():
    """Tests partition vector roundtrip."""
    labels = np.array([0, 0, 1, 2, 2])
    z = partition_vector_to_2d_matrix(labels)
    got = partition_matrix_to_vector(z)
    assert got.shape == labels.shape
    assert np.all(z == partition_vector_to_2d_matrix(got))


def test_contract_and_expand_shape():
    """Tests graph contraction and expansion."""
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


def test_contraction_maps_constraints_and_warm_start_partition():
    """Component mappings preserve valid pairs and partition relationships."""
    A = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=float,
    )
    A_small, node2comp = contract_adj_matrix_new(A, must_link=[(0, 1)])

    mapped = contract_node_pairs(
        [(0, 3), (1, 2), (3, 0)],
        node2comp,
        relation_name="cannot-link",
        reject_internal=True,
    )
    assert mapped == [(0, 1), (0, 2)]

    z_full = partition_vector_to_2d_matrix(np.array([0, 0, 1, 2]))
    z_small = contract_partition_matrix(z_full, node2comp)
    assert np.array_equal(z_small, np.eye(3, dtype=int))
    assert np.array_equal(expand_z_matrix(z_small, node2comp), z_full)
    assert compute_f_star(
        A,
        A.sum(axis=1),
        float(A.sum()),
        z_full,
    ) == pytest.approx(
        compute_f_star(
            A_small,
            A_small.sum(axis=1),
            float(A_small.sum()),
            z_small,
        )
    )


def test_contraction_rejects_conflicting_or_inconsistent_inputs():
    """Cannot-links and warm starts may not split a contracted component."""
    node2comp = np.array([0, 0, 1])

    with pytest.raises(ValueError, match="inside contracted component"):
        contract_node_pairs(
            [(0, 1)],
            node2comp,
            relation_name="cannot-link",
            reject_internal=True,
        )

    with pytest.raises(ValueError, match="not constant"):
        contract_partition_matrix(np.eye(3, dtype=int), node2comp)

    with pytest.raises(ValueError, match="outside 0..2"):
        contract_adj_matrix_new(
            np.zeros((3, 3)),
            must_link=[(0, 3)],
        )
