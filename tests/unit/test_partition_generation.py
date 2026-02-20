import numpy as np

from asunder.utils.partition_generation import make_partitions_random_links_only


def test_random_partitions_respect_shape():
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

