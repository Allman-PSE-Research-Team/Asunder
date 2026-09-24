import numpy as np

from asunder.base.utils.partition_generation import make_partitions_random_links_only


def test_random_partitions_respect_shape():
    """Tests that random partition generation respects shape"""
    parts = make_partitions_random_links_only(
        N=6,
        must_link=[(0, 1)],
        cannot_link=[(2, 3)],
        n_parts=3,
    )
    assert len(parts) >= 1
    for z in parts:
        assert z.shape == (6, 6)
        assert np.all(z == z.T)


def test_dsatur_partition_is_first_deterministic_pairwise_candidate():
    kwargs = {
        "N": 6,
        "must_link": [(0, 1)],
        "cannot_link": [(0, 2), (2, 3), (3, 4), (4, 0)],
        "n_parts": 1,
        "return_Z": False,
    }

    first = make_partitions_random_links_only(seed=1, **kwargs)[0]
    second = make_partitions_random_links_only(seed=999, **kwargs)[0]

    assert first["name"] == "dsatur"
    assert np.array_equal(first["g"], second["g"])
    assert first["g"][0] == first["g"][1]
    for i, j in kwargs["cannot_link"]:
        assert first["g"][i] != first["g"][j]


def test_dsatur_coloring_splits_classes_to_requested_K():
    result = make_partitions_random_links_only(
        N=5,
        K=3,
        n_parts=1,
        return_Z=False,
    )[0]

    assert result["name"] == "dsatur_fixedK"
    assert result["K_used"] == 3
    assert np.unique(result["g"]).size == 3

