import numpy as np

from asunder.evaluation.metrics import ari_sklearn, nmi_sklearn, permuted_accuracy, vi_sklearn


def test_metrics_identity():
    labels = np.array([0, 0, 1, 1, 2, 2])
    assert nmi_sklearn(labels, labels) == 1.0
    assert ari_sklearn(labels, labels) == 1.0
    assert vi_sklearn(labels, labels) == 0.0


def test_permuted_accuracy_vector_input():
    gt = np.array([0, 0, 1, 1])
    sol = np.array([1, 1, 0, 0])
    acc, mapping = permuted_accuracy(gt, sol)
    assert acc == 1.0
    assert mapping

