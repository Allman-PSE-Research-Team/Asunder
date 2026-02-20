import numpy as np

from asunder import CSDDecomposition, CSDDecompositionConfig
from asunder.column_generation.subproblem import heuristic_subproblem


def my_subproblem(A, a, m, duals, **kwargs):
    # Delegate to built-in heuristic with custom algorithm/package options.
    return heuristic_subproblem(A, a, m, duals, algo="louvain", package="networkx", **kwargs)


def main():
    A = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=float,
    )
    cfg = CSDDecompositionConfig(
        ifc_params={
            "generator": lambda N, **_: [np.ones((N, N), dtype=int)],
            "num": 1,
            "args": {"N": A.shape[0]},
        },
        extract_dual=False,
        final_master_solve=False,
        max_iterations=2,
        verbosity=0,
    )
    result = CSDDecomposition(config=cfg, subproblem_fn=my_subproblem).run(A)
    print("final objective:", result.final_master_obj)


if __name__ == "__main__":
    main()

