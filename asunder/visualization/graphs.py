"""Graph visualization helpers."""

from __future__ import annotations


def draw_network(G, community_map):
    """Draw a graph with node colors defined by an explicit community map."""
    import matplotlib.pyplot as plt
    import networkx as nx

    pos = nx.spring_layout(G, seed=42)
    cmap = plt.get_cmap("tab10", max(community_map.values()) + 1)
    nx.draw_networkx_nodes(
        G,
        pos,
        community_map.keys(),
        node_size=210,
        cmap=cmap,
        node_color=list(community_map.values()),
    )
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color="black", font_weight="bold")
    plt.show()


def draw_network_with_labels(
    G,
    community_map_labels,
    label=True,
    color_edges=False,
    legend=True,
    title=True,
    figsize=None,
):
    """Draw a community-colored graph with optional labels, legend, and edge styling."""
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    pos = nx.spring_layout(G, seed=42)
    unique_communities = sorted(set(community_map_labels.values()))
    num_communities = len(unique_communities)
    community_to_color_index = {comm: i for i, comm in enumerate(unique_communities)}

    if num_communities < 10:
        cmap = plt.get_cmap("tab10")
        colors = cmap(np.linspace(0, 1, cmap.N))
        if num_communities == 2:
            colors = ["#ffcb05", "#00274c"]
    else:
        colors = []
        for cmap_name in ["tab20", "tab20b", "tab20c"]:
            cmap = plt.get_cmap(cmap_name)
            colors.extend(cmap(np.linspace(0, 1, cmap.N)))

    node_colors = [community_to_color_index[community_map_labels[node]] for node in G.nodes()]
    node_color_values = [colors[idx] for idx in node_colors]
    if figsize:
        plt.figure(figsize=figsize)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_color_values,
        node_size=500 if num_communities < 10 else round(500 / num_communities, -1),
    )

    edges = G.edges(data=True)
    weights = [attr["weight"] for _, _, attr in edges]
    if color_edges:
        edge_colors = [community_to_color_index[community_map_labels[tgt]] for _, tgt, _ in edges]
        edge_color_values = [colors[idx] for idx in edge_colors]
        nx.draw_networkx_edges(
            G, pos, width=[0.5 + 0.7 * w for w in weights], edge_color=edge_color_values, alpha=0.6
        )
    else:
        nx.draw_networkx_edges(
            G,
            pos,
            width=[0.5 + 0.7 * w for w in weights],
            edge_color="#bfbfbf" if num_communities > 20 else "#000000",
            alpha=0.6,
        )

    if label and num_communities == 2:
        core = {k: k for k, v in community_map_labels.items() if v == 1}
        periphery = {k: k for k, v in community_map_labels.items() if v == 2}
        nx.draw_networkx_labels(G, pos, labels=periphery, font_size=8, font_color="white")
        nx.draw_networkx_labels(G, pos, labels=core, font_size=8, font_color="black")

    for comm, idx in community_to_color_index.items():
        plt.scatter([], [], color=colors[idx], label=f"Community {comm}")
    if legend:
        plt.legend(scatterpoints=1, frameon=True)
    if title:
        plt.title(
            "Constraint Graph with Core-Periphery Structure"
            if num_communities == 2
            else "Constraint Graph with Communities"
        )
    plt.axis("off")
    plt.show()


def draw_colored_constraint_graph(
    G,
    node_type: dict = None,
    edge_type: dict = None,
    color_nodes_by_type: bool = True,
    color_edges_by_type: bool = True,
    node_homog_color: str = "#00274C",
    edge_homog_color: str = "gray",
    linear_color: str = "#00274C",
    nonlinear_color: str = "#FFCB05",
    cont_color: str = "#FFCB05",
    int_color: str = "#00274C",
):
    """Render constraint graph with node/edge styling by linearity and variable type."""
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.lines import Line2D

    if color_nodes_by_type and node_type:
        node_colors = [
            nonlinear_color if node_type.get(n) == "nonlinear" else linear_color for n in G.nodes()
        ]
    else:
        node_colors = [node_homog_color] * G.number_of_nodes()

    if color_edges_by_type and edge_type:
        edge_colors = []
        for u, v in G.edges():
            key = (u, v) if (u, v) in edge_type else (v, u)
            et = edge_type.get(key, None)
            edge_colors.append(int_color if et == "integer" else cont_color)
    else:
        edge_colors = [edge_homog_color] * G.number_of_edges()

    pos = nx.spring_layout(G, k=0.2)
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(
        G, pos, node_color="white", edgecolors=node_colors, linewidths=2, node_size=500
    )
    integer_edges = [(u, v) for u, v, attr in G.edges(data=True) if attr["var_type"] == "integer"]
    continuous_edges = [(u, v) for u, v, attr in G.edges(data=True) if attr["var_type"] == "continuous"]
    nx.draw_networkx_edges(
        G, pos, edgelist=continuous_edges, style="solid", edge_color=cont_color, width=2
    )
    nx.draw_networkx_edges(
        G, pos, edgelist=integer_edges, style="--", edge_color=int_color, width=2
    )
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

    handles = []
    if color_nodes_by_type and node_type:
        handles += [
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor="white",
                markeredgecolor=linear_color,
                markersize=10,
                lw=0,
                label="Linear constraints",
            ),
            Line2D(
                [],
                [],
                marker="o",
                markerfacecolor="white",
                markeredgecolor=nonlinear_color,
                markersize=10,
                lw=0,
                label="Nonlinear constraints",
            ),
        ]
    if color_edges_by_type and edge_type:
        handles += [
            Line2D([], [], color=cont_color, lw=2, label="Continuous variables"),
            Line2D([], [], color=int_color, lw=2, linestyle="--", label="Integer variables"),
        ]
    if handles:
        plt.legend(handles=handles, loc="upper right")
    plt.axis("off")
    plt.show()
