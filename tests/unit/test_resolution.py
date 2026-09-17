import numpy as np
import pytest

from asunder.base.column_generation import subproblem as subproblem_module
from asunder.base.column_generation.subproblem import (
    custom_heuristic_subproblem,
    heuristic_subproblem,
)


def test_unsupported_backend_rejects_nondefault_resolution_before_dispatch():
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="does not support"):
        heuristic_subproblem(
            adjacency,
            adjacency.sum(axis=1),
            float(adjacency.sum()),
            {},
            algo="girvan_newman",
            package="networkx",
            gamma=1.25,
        )


def test_custom_pricing_rejects_nondefault_resolution():
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="does not support"):
        custom_heuristic_subproblem(
            adjacency,
            adjacency.sum(axis=1),
            float(adjacency.sum()),
            {},
            algo="spectral",
            gamma=1.25,
        )


def test_modified_louvain_receives_nondefault_resolution(monkeypatch):
    adjacency = np.array([[0.0, 1.0], [1.0, 0.0]])
    observed = {}

    class FakeModifiedLouvain:
        def __init__(
            self,
            *,
            resolution,
            random_state,
            max_dense_working_bytes,
        ):
            observed["resolution"] = resolution
            observed["seed"] = random_state
            observed["max_dense_working_bytes"] = max_dense_working_bytes

        def fit(self, A, duals):
            self.labels_ = np.array([0, 1])
            self.obj_val_ = 0.0

    monkeypatch.setattr(
        subproblem_module,
        "ModifiedLouvain",
        FakeModifiedLouvain,
    )
    custom_heuristic_subproblem(
        adjacency,
        adjacency.sum(axis=1),
        float(adjacency.sum()),
        {},
        algo="full_louvain",
        gamma=1.25,
        seed=7,
    )

    assert observed == {
        "resolution": 1.25,
        "seed": 7,
        "max_dense_working_bytes": 512 * 1024**2,
    }
