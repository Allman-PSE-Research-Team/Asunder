import numpy as np
import pytest

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
    weights = np.array([[0.0, 10.0], [10.0, 0.0]])

    quantized, _ = quantize_metis_weights(weights, safe_total=1_000)

    assert int(quantized.sum()) <= 1_000
    assert quantized[0, 1] == quantized[1, 0] == 500


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
