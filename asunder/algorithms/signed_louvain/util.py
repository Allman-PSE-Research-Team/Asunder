#!/usr/bin/env python
# coding: utf-8
# Source: https://github.com/lejohnyjohn/signed-louvain/blob/main/util.py
# Description: Taken from repo on July 18, 2025

# In[ ]:

"""Utility helpers for constructing signed Louvain input graphs."""


import networkx as nx

# In[ ]:


def build_nx_graph(n, edge_list):
    """
    input
        n int: number of nodes in the graph
        edge_list list: list of (u, v, s) where u, v are integers indexed from 0 to n-1, 
                        and s is a weight
    returns
        graph: nx.Graph object 
    """
    graph = nx.Graph()
    graph.add_nodes_from(list(range(n)))
    edges = [(edge[0], edge[1], {'weight': edge[2]}) for edge in edge_list]
    graph.add_edges_from(edges)
    
    return graph


# In[ ]:


def build_subgraphs(G, weight='weight'):
    """ Build positive and negatives graphs. """
    pos_edges = [edge for edge in G.edges(data=True) if edge[2][weight] > 0]
    posG = nx.Graph()
    posG.add_nodes_from(G.nodes())
    posG.add_edges_from(pos_edges)
    
    neg_edges = [(edge[0], 
                  edge[1], 
                  {weight: -edge[2][weight]}
                 ) for edge in G.edges(data=True) if edge[2][weight] < 0
                ]
    negG = nx.Graph()
    negG.add_nodes_from(G.nodes())
    negG.add_edges_from(neg_edges)
    
    return posG, negG

