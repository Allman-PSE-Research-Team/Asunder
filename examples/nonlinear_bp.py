import networkx as nx

from asunder.nlbnp import NonlinearBranchAndPrice


def main():
    graph = nx.Graph()
    graph.add_nodes_from(
        [
            ("n1", {"kind": "nonlinear"}),
            ("n2", {"kind": "linear"}),
            ("n3", {"kind": "linear"}),
            ("n4", {"kind": "linear"}),
        ]
    )
    graph.add_edge("n1", "n2", relationship="integer")
    graph.add_edge("n2", "n3", relationship="continuous")
    graph.add_edge("n3", "n4", relationship="continuous")
    graph.add_edge("n1", "n4", relationship="integer")

    result = NonlinearBranchAndPrice(
        graph,
        worthy_edge_attr="relationship",
        worthy_edge_value="integer",
        nonlinear_node_attr="kind",
        nonlinear_node_value="nonlinear",
        final_master_solve=False,
        max_iterations=3,
        disable_tqdm=True,
    )
    if result.final_partition is None:
        raise RuntimeError(result.metadata)
    print(result.metadata["community_map_labels"])
    print("maximum linear-only cardinality:", result.metadata["K_max"])


if __name__ == "__main__":
    main()
