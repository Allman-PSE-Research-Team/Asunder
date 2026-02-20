from asunder.algorithms.signed_louvain.community_detection import best_partition
from asunder.algorithms.signed_louvain.util import build_nx_graph, build_subgraphs


def test_build_nx_graph_and_subgraphs():
    g = build_nx_graph(3, [(0, 1, 1.0), (1, 2, -2.0)])
    pos, neg = build_subgraphs(g)
    assert pos.number_of_edges() == 1
    assert neg.number_of_edges() == 1


def test_signed_louvain_best_partition_smoke():
    g = build_nx_graph(4, [(0, 1, 1.0), (2, 3, 1.0), (1, 2, -0.5)])
    pos, neg = build_subgraphs(g)
    part, status = best_partition(
        layers=[pos, neg],
        layer_weights=[1.0, -1.0],
        resolutions=[1.0, 1.0],
        masks=[False, True],
        k=2,
        pass_max=5,
        silent=True,
    )
    assert isinstance(part, dict)
    assert hasattr(status, "modularity")

