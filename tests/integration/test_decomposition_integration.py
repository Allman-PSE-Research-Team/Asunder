import numpy as np

from asunder import CSDDecomposition, CSDDecompositionConfig


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

