import numpy as np
from scipy import sparse

from asunder import CSDDecomposition, CSDDecompositionConfig
from asunder.base.column_generation.decomposition import CSD_decomposition


def _ifc_generator(N, **_):
    return [np.ones((N, N), dtype=int)]


def _master(A, a, m, Z_star, f_stars, **kwargs):
    return [1.0] + [0.0] * (len(Z_star) - 1), {"mu_dual": 0.0}, float(f_stars[0] if f_stars else 0.0)


def _subproblem(A, a, m, duals, **kwargs):
    return 0.0, np.eye(A.shape[0], dtype=int)


def test_custom_master_and_subproblem_wiring():
    """Tests custom master and subproblem wiring."""
    A = np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=float,
    )
    cfg = CSDDecompositionConfig(
        ifc_params={"generator": _ifc_generator, "num": 1, "args": {"N": A.shape[0]}},
        final_master_solve=False,
        max_iterations=3,
        tolerance=1e-8,
        verbose=0,
    )
    result = CSDDecomposition(config=cfg, master_fn=_master, subproblem_fn=_subproblem).run(A)
    assert result.records
    assert result.final_partition is not None


def test_resolution_is_forwarded_to_pricing():
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])
    observed = {}

    def resolution_subproblem(A, a, m, duals, *, gamma):
        observed["gamma"] = gamma
        return 0.0, np.eye(A.shape[0], dtype=int)

    config = CSDDecompositionConfig(
        resolution=1.25,
        ifc_params={
            "generator": _ifc_generator,
            "num": 1,
            "args": {"N": adjacency.shape[0]},
        },
        use_refined_column=False,
        refine_post_loop=False,
        final_master_solve=False,
        max_iterations=1,
        disable_tqdm=True,
        verbose=0,
    )

    result = CSDDecomposition(
        config=config,
        master_fn=_master,
        subproblem_fn=resolution_subproblem,
    ).run(adjacency)

    assert observed["gamma"] == 1.25
    assert result.metadata["resolution"] == 1.25


def test_csd_preserves_sparse_adjacency_and_csr_columns():
    adjacency = sparse.csr_matrix(
        np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    )
    observed = {}

    def sparse_master(A, a, m, Z_star, f_stars, **kwargs):
        observed["adjacency"] = A
        observed["column"] = Z_star[0]
        return [1.0], {"mu_dual": 0.0}, float(f_stars[0])

    config = CSDDecompositionConfig(
        column_storage="csr",
        ifc_params={
            "generator": _ifc_generator,
            "num": 1,
            "args": {"N": adjacency.shape[0]},
        },
        refine_post_loop=False,
        final_master_solve=False,
        max_iterations=1,
        disable_tqdm=True,
        verbose=0,
    )
    result = CSDDecomposition(
        config=config,
        master_fn=sparse_master,
        subproblem_fn=_subproblem,
    ).run(adjacency)

    assert sparse.isspmatrix_csr(observed["adjacency"])
    assert sparse.isspmatrix_csr(observed["column"])
    assert sparse.isspmatrix_csr(result.final_partition)
    assert result.metadata["final_partition_storage"] == "csr"


def test_mixed_post_loop_columns_fall_back_to_csr_and_report_final_pool():
    """Oversized mixed aggregation remains sparse and metadata describes the final pool."""
    adjacency = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    )
    strengths = adjacency.sum(axis=1)
    dense_column = np.ones((3, 3), dtype=bool)
    csr_column = sparse.eye(3, format="csr", dtype=bool)
    observed = {}

    def fractional_master(A, a, m, Z_star, f_stars, **kwargs):
        return [0.5, 0.5], {"mu_dual": 0.0}, 0.0

    def capture_refiner(A, partition, **kwargs):
        observed["partition"] = partition
        return dense_column

    raw = CSD_decomposition(
        adjacency,
        strengths,
        float(strengths.sum()),
        fractional_master,
        _subproblem,
        columns=[dense_column, csr_column],
        f_stars=[0.0, 0.0],
        refine_params={"refine_func": capture_refiner},
        refine_post_loop=True,
        final_master_solve=False,
        max_iterations=1,
        disable_tqdm=True,
        verbose=-1,
        column_storage="auto",
        sparse_column_density_threshold=0.5,
        max_dense_working_bytes=1,
    )

    assert sparse.isspmatrix_csr(observed["partition"])
    assert len(raw) == 2
    assert "storage_metadata" not in raw[0]
    assert raw[-1]["storage_metadata"]["column_storage_counts"] == {
        "dense": 2,
        "csr": 1,
    }

