import numpy as np
import pytest

from asunder.base.algorithms.core_periphery import (
    CorePeripheryResult,
    FullContinuousGeneticBE,
    contract_core_periphery_adjacency,
    detect_continuous_KL,
    normalized_BE_score,
    spectral_continuous_cp_detection,
)
from asunder.nlbnp.algorithms.core_periphery import _detect_core_periphery


def test_aggregate_contraction_preserves_block_strength_and_internal_diagonal():
    A = np.array(
        [
            [0.0, 2.0, 3.0, 0.0],
            [2.0, 0.0, 1.0, 4.0],
            [3.0, 1.0, 0.0, 5.0],
            [0.0, 4.0, 5.0, 0.0],
        ]
    )

    contraction = contract_core_periphery_adjacency(A, must_link=[(0, 1)])

    expected_strength = np.array([A[[0, 1]].sum(), A[2].sum(), A[3].sum()])
    assert np.allclose(contraction.adjacency.sum(axis=1), expected_strength)
    assert contraction.adjacency[0, 0] == 2 * A[0, 1]
    assert contraction.block_sizes.tolist() == [2.0, 1.0, 1.0]


def test_contraction_rejects_asymmetric_adjacency():
    with pytest.raises(ValueError, match="symmetric"):
        contract_core_periphery_adjacency(np.array([[0.0, 1.0], [0.0, 0.0]]))


def test_singleton_blocks_make_ordinary_and_generalized_spectral_equal():
    A = np.array(
        [
            [0.0, 2.0, 1.0],
            [2.0, 0.0, 3.0],
            [1.0, 3.0, 0.0],
        ]
    )

    ordinary = spectral_continuous_cp_detection(A, target="contracted", normalize=False)
    generalized = spectral_continuous_cp_detection(A, target="original", normalize=False)

    assert ordinary.eigenproblem == "ordinary"
    assert generalized.eigenproblem == "generalized"
    assert np.allclose(ordinary.eigenvalues, generalized.eigenvalues)
    assert np.allclose(
        np.abs(ordinary.block_eigenvectors),
        np.abs(generalized.block_eigenvectors),
    )


def test_equal_block_sizes_make_generalized_solution_equivalent_up_to_scaling():
    A = np.array(
        [
            [0, 1, 4, 2, 0, 1],
            [1, 0, 3, 1, 1, 0],
            [4, 3, 0, 2, 2, 1],
            [2, 1, 2, 0, 1, 2],
            [0, 1, 2, 1, 0, 3],
            [1, 0, 1, 2, 3, 0],
        ],
        dtype=float,
    )
    pairs = [(0, 1), (2, 3), (4, 5)]

    ordinary = spectral_continuous_cp_detection(
        A,
        must_link=pairs,
        target="contracted",
        normalize=False,
    )
    generalized = spectral_continuous_cp_detection(
        A,
        must_link=pairs,
        target="original",
        normalize=False,
    )

    assert np.allclose(ordinary.eigenvalues / 2.0, generalized.eigenvalues)
    assert np.allclose(
        np.abs(ordinary.block_eigenvectors) / np.sqrt(2.0),
        np.abs(generalized.block_eigenvectors),
    )


def test_generalized_spectral_solution_satisfies_residual_for_unequal_blocks():
    A = np.array(
        [
            [0, 2, 1, 4],
            [2, 0, 3, 1],
            [1, 3, 0, 2],
            [4, 1, 2, 0],
        ],
        dtype=float,
    )
    result = spectral_continuous_cp_detection(
        A,
        must_link=[(0, 1), (1, 2)],
        target="original",
        normalize=False,
    )
    g = result.block_eigenvectors[:, 0]
    residual = result.contracted_adjacency @ g - result.eigenvalues[0] * result.block_sizes * g

    assert np.linalg.norm(residual) < 1e-10
    assert result.eigenproblem == "generalized"


@pytest.mark.parametrize("target", ["contracted", "original"])
def test_rank_two_uses_scaled_largest_magnitude_embedding(target):
    A = np.array(
        [
            [0, 5, 4, 4],
            [5, 0, 3, 3],
            [4, 3, 0, 0],
            [4, 3, 0, 0],
        ],
        dtype=float,
    )

    result = spectral_continuous_cp_detection(A, target=target, spectral_rank=2)

    assert result.block_embedding.shape == (4, 2)
    assert np.all(np.abs(result.eigenvalues[:-1]) >= np.abs(result.eigenvalues[1:]))
    assert np.allclose(
        result.block_embedding,
        result.block_eigenvectors * np.sqrt(np.abs(result.eigenvalues))[None, :],
    )


@pytest.mark.parametrize("algorithm", ["GA", "KL"])
@pytest.mark.parametrize("target", ["contracted", "original"])
def test_ga_and_kl_return_typed_target_aware_block_results(algorithm, target):
    A = np.array(
        [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 0.0],
        ]
    )
    if algorithm == "GA":
        result = FullContinuousGeneticBE(
            A,
            must_link=[(0, 1)],
            must_group=[2, 3],
            target=target,
            pop_size=6,
            generations=1,
            tournament_size=2,
            seed=7,
        ).run()
    else:
        result = detect_continuous_KL(
            A,
            must_link=[(0, 1)],
            must_group=[2, 3],
            target=target,
            max_iter=1,
            seed=7,
        )

    assert isinstance(result, CorePeripheryResult)
    assert result.target_space == target
    assert result.block_scores.shape == (2,)
    assert result.continuous_node_scores[0] == result.continuous_node_scores[1]
    assert result.continuous_node_scores[2] == result.continuous_node_scores[3]


@pytest.mark.parametrize("target", ["contracted", "original"])
def test_continuous_fits_use_the_declared_pair_samples(target):
    A = np.array(
        [
            [0.0, 3.0, 0.0, 2.0],
            [3.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 4.0],
            [2.0, 0.0, 4.0, 0.0],
        ]
    )
    result = detect_continuous_KL(
        A,
        must_link=[(0, 1)],
        target=target,
        max_iter=0,
        seed=3,
    )
    scores = result.block_scores
    original_reconstruction = np.outer(scores, scores)
    contracted_reconstruction = original_reconstruction
    if target == "original":
        contracted_reconstruction = (
            result.block_sizes[:, None]
            * original_reconstruction
            * result.block_sizes[None, :]
        )

    block_i, block_j = np.triu_indices(len(result.blocks), k=0)
    node_i, node_j = np.triu_indices(A.shape[0], k=1)
    expanded_scores = scores[result.node_to_block]
    expected_contracted = np.corrcoef(
        result.contracted_adjacency[block_i, block_j],
        contracted_reconstruction[block_i, block_j],
    )[0, 1]
    expected_original = np.corrcoef(
        A[node_i, node_j],
        (expanded_scores[:, None] * expanded_scores[None, :])[node_i, node_j],
    )[0, 1]

    assert result.contracted_fit == pytest.approx(expected_contracted)
    assert result.original_fit == pytest.approx(expected_original)


def test_detection_names_primary_and_diagnostic_fits_in_selected_spaces():
    A = np.array(
        [
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )

    contracted = _detect_core_periphery(
        A,
        must_link=[(0, 1)],
        target="contracted",
        prob_method="threshold",
        threshold=0.5,
    )
    original = _detect_core_periphery(
        A,
        must_link=[(0, 1)],
        target="original",
        prob_method="threshold",
        threshold=0.5,
    )

    assert contracted.primary_fit == contracted.contracted_fit
    assert original.primary_fit == original.original_fit
    assert contracted.contracted_be_score == normalized_BE_score(
        contracted.contracted_adjacency,
        contracted.block_labels,
        include_diagonal=True,
    )
    assert original.original_be_score == normalized_BE_score(A, original.node_labels)


def test_rank_two_detection_requires_gaussian_mixture():
    A = np.ones((3, 3), dtype=float) - np.eye(3)

    with pytest.raises(ValueError, match="gaussian_mixture"):
        _detect_core_periphery(
            A,
            spectral_rank=2,
            prob_method="threshold",
        )
