import networkx as nx
import numpy as np

from asunder import CSDDecomposition, CSDDecompositionConfig
from asunder.case_studies import build_circle_cutting_graph


def main():
    G, _, _ = build_circle_cutting_graph(
        num_circles=5,
        num_rectangles=4,
        dimensions=["x", "y"],
    )
    A = nx.to_numpy_array(G, dtype=float)

    cfg = CSDDecompositionConfig(
        ifc_params={
            "generator": lambda N, **_: [np.ones((N, N), dtype=int)],
            "num": 1,
            "args": {"N": A.shape[0]},
        },
        extract_dual=False,
        final_master_solve=False,
        max_iterations=3,
        verbosity=0,
    )
    result = CSDDecomposition(config=cfg).run(A)
    print("iterations:", result.metadata["n_iterations"])


if __name__ == "__main__":
    main()
