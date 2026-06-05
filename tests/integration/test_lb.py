import networkx as nx
import numpy as np
import pytest
from test_case_studies_small_instances import _configure_solver_or_skip

from asunder.base.utils.graph import get_optimization_params_from_graph
from asunder.load_balancing import LoadBalancer
from asunder.load_balancing.algorithms.VFD import very_fortunate_descent


# Toy Problem
def cliques_with_bridges(clique_sizes=(12, 7, 5, 4), extra_bridges=2, seed=0):
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    offset = 0
    cliques = []

    # build disjoint cliques on consecutive node labels
    for s in clique_sizes:
        nodes = list(range(offset, offset + s))
        G.add_nodes_from(nodes)
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                G.add_edge(nodes[i], nodes[j])
        cliques.append(nodes)
        offset += s

    # connect cliques in a chain (guarantees connected)
    for k in range(len(cliques) - 1):
        u = rng.choice(cliques[k])
        v = rng.choice(cliques[k+1])
        G.add_edge(int(u), int(v))

    # add a few extra random inter-clique bridges
    for _ in range(extra_bridges):
        a, b = rng.choice(len(cliques), size=2, replace=False)
        u = rng.choice(cliques[a])
        v = rng.choice(cliques[b])
        G.add_edge(int(u), int(v))

    # ground-truth clique id per node
    g = np.empty(G.number_of_nodes(), dtype=int)
    for cid, nodes in enumerate(cliques):
        g[nodes] = cid

    return G, g

@pytest.mark.solver
def test_load_balancing_toy_problem():
    """Tests load balancing workflow."""
    _configure_solver_or_skip()

    G, _ = cliques_with_bridges((12,7,5), extra_bridges=1)

    result = LoadBalancer(G, K=3, R=0, ifc_generator="ordered", seed=42, disable_tqdm=True)
    z = result.final_partition
    
    assert z.sum(axis=1).max() == z.sum(axis=1).min()

def test_vfd_load_balancing_toy_problem():
    """Tests local search refinement algorithm (VFD)."""

    G, _ = cliques_with_bridges((12,7,5), extra_bridges=1)

    A, a, m = get_optimization_params_from_graph(G=G)

    out = very_fortunate_descent(
        wz=A, A=A, a=a, m=m,
        K=3, R=0,
        seed=42,
        shake_rounds=1
    )

    assert out is not None

    z, _ = out
    
    assert z.sum(axis=1).max() == z.sum(axis=1).min()
