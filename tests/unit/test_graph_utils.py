import tracemalloc

import numpy as np
import pytest
from scipy import sparse

from asunder.base.column_generation.master import compute_f_star
from asunder.base.utils.graph import (
    contract_adj_matrix_new,
    contract_node_pairs,
    contract_partition_matrix,
    expand_z_matrix,
    partition_matrix_to_vector,
    partition_vector_to_2d_matrix,
    validate_partition_matrix,
    z_hamming_upper,
)
from asunder.base.utils.matrix import checked_to_dense


def test_partition_vector_roundtrip():
    """Tests partition vector roundtrip."""
    labels = np.array([0, 0, 1, 2, 2])
    z = partition_vector_to_2d_matrix(labels)
    got = partition_matrix_to_vector(z)
    assert got.shape == labels.shape
    assert np.all(z == partition_vector_to_2d_matrix(got))


def test_sparse_partition_storage_roundtrip_and_objective_equivalence():
    labels = np.arange(16)
    dense = partition_vector_to_2d_matrix(labels, storage="dense")
    csr = partition_vector_to_2d_matrix(labels, storage="auto")

    assert dense.dtype == np.bool_
    assert dense.astype(np.int64).nbytes == 8 * dense.nbytes
    assert sparse.isspmatrix_csr(csr)
    assert sparse.isspmatrix_csr(validate_partition_matrix(csr))
    assert np.array_equal(partition_matrix_to_vector(csr), labels)
    assert z_hamming_upper(dense, csr) == 0.0
    merged = partition_vector_to_2d_matrix(np.zeros(labels.size))
    assert z_hamming_upper(merged, csr) == z_hamming_upper(csr, merged) == 1.0
    assert z_hamming_upper(merged, dense) == 1.0
    assert validate_partition_matrix(dense, storage="auto") is dense

    adjacency = sparse.csr_matrix(
        np.equal.outer(labels, np.roll(labels, 1)).astype(float)
    )
    adjacency = adjacency + adjacency.T
    strengths = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    assert compute_f_star(adjacency, strengths, float(strengths.sum()), csr) == pytest.approx(
        compute_f_star(adjacency.toarray(), strengths, float(strengths.sum()), dense)
    )

    grouped = partition_vector_to_2d_matrix(labels // 4)
    fractional = 0.25 * grouped + 0.75 * dense
    with pytest.raises(ValueError, match="square matrices"):
        compute_f_star(np.ones((2, 3)), np.ones(3), 6.0, np.ones((2, 3)))
    for dtype in (bool, np.uint8, np.float32, np.float64):
        weights = (200 * adjacency.toarray()).astype(dtype)
        strengths = weights.sum(axis=1, dtype=np.float64)
        volume = float(strengths.sum())
        objective = weights.astype(float) / volume - 1.7 * np.outer(strengths, strengths) / volume**2
        for partition in (grouped, fractional):
            expected = float(np.sum(objective * partition))
            for graph in (weights, sparse.csr_matrix(weights)):
                for column in (partition, sparse.csr_matrix(partition)):
                    assert compute_f_star(graph, strengths, volume, column, gamma=1.7) == pytest.approx(expected)

    # A small allocation regression: either dense term used to allocate a full
    # float64 matrix. Row-wise scoring should stay well below even N**2 bytes.
    n_nodes = 256
    dense_graph = np.eye(n_nodes)
    dense_partition = np.eye(n_nodes, dtype=bool)
    strengths = np.ones(n_nodes)
    for graph in (dense_graph, sparse.csr_matrix(dense_graph)):
        tracemalloc.start()
        try:
            score = compute_f_star(graph, strengths, float(n_nodes), dense_partition)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert score == pytest.approx(1 - 1 / n_nodes)
        assert peak < n_nodes**2

    malformed = sparse.eye(3, format="lil", dtype=bool)
    malformed[0, 1] = malformed[1, 0] = True
    malformed[1, 2] = malformed[2, 1] = True
    with pytest.raises(ValueError, match="transitive"):
        validate_partition_matrix(malformed.tocsr())


def test_checked_dense_conversion_uses_working_set_cap(monkeypatch):
    matrix = sparse.eye(10, format="csr")
    with pytest.raises(MemoryError, match="estimated.*working set"):
        checked_to_dense(
            matrix,
            working_arrays=2,
            max_dense_working_bytes=100,
            operation="test conversion",
        )
    dense = matrix.toarray()
    assert checked_to_dense(dense, max_dense_working_bytes=1) is dense
    with pytest.raises(MemoryError):
        checked_to_dense(dense, working_arrays=2, max_dense_working_bytes=100)
    with pytest.raises(MemoryError):
        checked_to_dense(dense, dtype=bool, max_dense_working_bytes=50)
    original_toarray = sparse.csr_matrix.toarray
    dtypes = []

    def record_dtype(self, *args, **kwargs):
        dtypes.append(self.dtype)
        return original_toarray(self, *args, **kwargs)

    monkeypatch.setattr(sparse.csr_matrix, "toarray", record_dtype)
    converted = checked_to_dense(matrix, dtype=bool, max_dense_working_bytes=100)
    assert converted.dtype == bool
    assert dtypes == [np.dtype(bool)]
    for cap in (1600, None):
        assert np.array_equal(checked_to_dense(matrix, working_arrays=2, max_dense_working_bytes=cap), dense)


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
    with pytest.raises(MemoryError, match="original-size partition expansion"):
        expand_z_matrix(z_small, node2comp, max_dense_working_bytes=100)
    with pytest.raises(MemoryError, match="adjacency contraction"):
        contract_adj_matrix_new(A, must_link=[(0, 1)], max_dense_working_bytes=100)
    for cap in (4096, None):
        assert np.array_equal(expand_z_matrix(z_small, node2comp, max_dense_working_bytes=cap), z_full)
        contracted, _ = contract_adj_matrix_new(A, must_link=[(0, 1)], max_dense_working_bytes=cap)
        assert np.array_equal(contracted, A_sup)
    assert sparse.issparse(expand_z_matrix(sparse.csr_matrix(z_small), node2comp, max_dense_working_bytes=1))


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


def test_contraction_preserves_existing_diagonal_mass_exactly():
    A = np.array(
        [
            [2.0, 3.0, 0.0],
            [3.0, 4.0, 1.0],
            [0.0, 1.0, 5.0],
        ]
    )
    contracted, node2comp = contract_adj_matrix_new(A, must_link=[(0, 1)])

    assert node2comp.tolist() == [0, 0, 1]
    assert np.array_equal(contracted, np.array([[12.0, 1.0], [1.0, 5.0]]))
    assert np.array_equal(contracted.sum(axis=1), np.array([13.0, 6.0]))


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
