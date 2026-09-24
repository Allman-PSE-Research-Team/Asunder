import ctypes

import numpy as np
import pytest

from asunder.load_balancing.algorithms import qmetis as qmetis_module
from asunder.load_balancing.algorithms._qmetis_wrapper import (
    METIS_OBJTYPE_MOD,
    METIS_OK,
    METIS_OPTION_CONTIG,
    METIS_OPTION_MODRESOLUTION,
    METIS_OPTION_NITER,
    METIS_OPTION_OBJTYPE,
    METIS_OPTION_SEED,
    QMETISBinding,
)
from asunder.load_balancing.algorithms.qmetis import (
    QMETISApproximationWarning,
    _epsilon_to_ubvec,
    _integer_weight_graph,
    quantize_metis_weights,
)


def test_quantize_metis_weights_preserves_symmetry_and_relative_order():
    weights = np.array(
        [
            [9.0, 0.1250001, 2.5],
            [0.1249999, 3.0, 1.75],
            [2.5, 1.75, 4.0],
        ]
    )

    with pytest.warns(QMETISApproximationWarning, match="nonzero diagonal"):
        quantized, scale = quantize_metis_weights(weights)

    assert quantized.dtype == np.int64
    assert np.array_equal(quantized, quantized.T)
    assert np.all(np.diag(quantized) == 0)
    assert quantized[0, 2] > quantized[1, 2] > quantized[0, 1] > 0
    assert scale > 0


def test_quantize_metis_weights_honors_accumulated_weight_budget():
    weights = np.array([[0.0, 1_000.0], [1_000.0, 0.0]])

    quantized, _ = quantize_metis_weights(weights, safe_total=1_000)

    assert int(quantized.sum()) <= 1_000
    assert quantized[0, 1] == quantized[1, 0] == 500


def test_quantize_metis_weights_preserves_safe_integer_valued_floats():
    weights = np.array([[0.0, 2.0, 5.0], [2.0, 0.0, 3.0], [5.0, 3.0, 0.0]])

    quantized, scale = quantize_metis_weights(weights)

    assert np.array_equal(quantized, weights.astype(np.int64))
    assert scale == 1.0


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (np.ones((2, 3)), "square"),
        (np.array([[0.0, np.nan], [np.nan, 0.0]]), "NaN"),
        (np.array([[0.0, -1.0], [-1.0, 0.0]]), "nonnegative"),
        (np.zeros((2, 2)), "positive edge"),
    ],
)
def test_quantize_metis_weights_rejects_invalid_input(weights, message):
    with pytest.raises(ValueError, match=message):
        quantize_metis_weights(weights)


def test_integer_graph_retains_python_integer_weights():
    adjacency = np.array([[0, 2**54], [2**54, 0]], dtype=np.int64)

    graph = _integer_weight_graph(adjacency)

    assert graph[0][1]["weight"] == 2**54
    assert isinstance(graph[0][1]["weight"], int)


def test_quantization_drops_diagonal_before_testing_for_positive_edges():
    weights = np.diag([4.0, 3.0])

    with pytest.warns(QMETISApproximationWarning, match="nonzero diagonal"):
        with pytest.raises(ValueError, match="positive edge"):
            quantize_metis_weights(weights)


def test_zero_epsilon_requests_exact_upper_balance_factor():
    assert _epsilon_to_ubvec(0.0, None) == [1.0]


def test_internal_wrapper_uses_qmetis_option_layout_and_fixed_resolution():
    binding = object.__new__(QMETISBinding)
    binding.idx_t = ctypes.c_int64

    def set_defaults(options):
        for index in range(len(options)):
            options[index] = -1
        return METIS_OK

    binding._set_default_options = set_defaults
    options = binding._make_options(
        resolution=1.25,
        supplied={"niter": 20, "seed": 7, "contig": True},
    )

    assert options[METIS_OPTION_OBJTYPE] == METIS_OBJTYPE_MOD
    assert options[METIS_OPTION_MODRESOLUTION] == 1_250_000
    assert options[METIS_OPTION_NITER] == 20
    assert options[METIS_OPTION_SEED] == 7
    assert options[METIS_OPTION_CONTIG] == 1


def test_internal_wrapper_rejects_non_modularity_objective():
    binding = object.__new__(QMETISBinding)
    binding.idx_t = ctypes.c_int64
    binding._set_default_options = lambda options: METIS_OK

    with pytest.raises(ValueError, match="only the modularity"):
        binding._make_options(
            resolution=1.0,
            supplied={"objtype": "cut"},
        )


def test_run_qmetis_forwards_resolution_without_rescaling_objective(monkeypatch):
    observed = {}

    def fake_partition(graph, **kwargs):
        observed.update(kwargs)
        return {"partition": [0, 0, 1], "obj_val": 0.375}

    monkeypatch.setattr(
        qmetis_module,
        "qmetis_load_balanced_partition",
        fake_partition,
    )
    adjacency = np.array(
        [
            [0.0, 2.0, 0.0],
            [2.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )

    partition, objective = qmetis_module.run_qmetis(
        adjacency,
        2,
        resolution=1.25,
    )

    assert observed["resolution"] == pytest.approx(1.25)
    assert objective == pytest.approx(0.375)
    assert np.array_equal(
        partition,
        np.equal.outer([0, 0, 1], [0, 0, 1]).astype(int),
    )
