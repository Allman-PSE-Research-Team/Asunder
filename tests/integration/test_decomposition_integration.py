import numpy as np

from asunder import CSDDecomposition, CSDDecompositionConfig


def _ifc_generator(N, **_):
    return [np.ones((N, N), dtype=int)]


def _master(A, a, m, Z_star, f_stars, **kwargs):
    return [1.0] + [0.0] * (len(Z_star) - 1), {"mu_dual": 0.0}, float(f_stars[0] if f_stars else 0.0)


def _subproblem(A, a, m, duals, **kwargs):
    return 0.0, np.eye(A.shape[0], dtype=int)


def test_custom_master_and_subproblem_wiring():
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
        extract_dual=True,
        final_master_solve=False,
        max_iterations=3,
        tolerance=1e-8,
        verbosity=0,
    )
    result = CSDDecomposition(config=cfg, master_fn=_master, subproblem_fn=_subproblem).run(A)
    assert result.records
    assert result.final_partition is not None

