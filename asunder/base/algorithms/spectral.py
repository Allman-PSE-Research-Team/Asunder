"""Spectral subproblem and matrix-based refinement routines."""
from __future__ import annotations

import networkx as nx
import numpy as np

from asunder.base.utils.graph import group_nodes_by_community


def matrix_to_group_vector(P):
    """
    Converts a partition matrix P into a group assignment vector s.
    
    Parameters
    ----------
    P : np.ndarray of int, shape (N, N)
        Input partition.
    
    Returns
    -------
    s : np.ndarray of int, shape (N,)
        group assignment vector
    """
    N = P.shape[0]
    s = np.ones(N, dtype=int)
    # Use the first row as a reference: if node i is in the same community as node 0 then assign 1, else -1.
    for i in range(N):
        s[i] = 1 if P[0, i] == 1 else -1
    return s

def group_vector_to_matrix(s):
    """
    Converts a group assignment vector s (with entries 1 or -1) into a partition matrix P.
    
    Parameters
    ----------
    s : np.ndarray of int, shape (N,)
        Group assignment vector.
    
    Returns
    -------
    P : np.ndarray of int, shape (N, N)
        Input partition.
    """
    s = s.flatten()
    N = len(s)
    # Compute outer product: if s[i]==s[j] then s[i]*s[j] = 1, so (1+1)/2 = 1; else (-1+1)/2 = 0.
    P = (np.outer(s, s) + np.ones((N, N), dtype=int)) // 2
    return P

def vector_to_communities(s):
    """
    Converts a group vector s into a list of communities (each community is a set of node indices).
    
    Parameters
    ----------
    s : np.ndarray of int, shape (N,)
        Group assignment vector.
    
    Returns
    -------
    communities: list[set[int]]
        List of communities.
    """
    community1 = set(np.where(s == 1)[0])
    community2 = set(np.where(s == -1)[0])
    communities = []
    if community1:
        communities.append(community1)
    if community2:
        communities.append(community2)
    return communities

def compute_modularity_from_vector(B_g, s):
    """
    Computes modularity of graph G for the partition described by vector s.
    
    Parameters
    ----------
    B_g : np.ndarray
        Subset of modularity matrix.
    s : np.ndarray of int, shape (N,)
        Group assignment vector.
    
    Returns
    -------
    float
        Modularity of the network subset.
    """
    z = group_vector_to_matrix(s)
    return np.sum(B_g * z)

def flip_vertex(s, vertex):
    """
    Returns a new group vector after flipping the sign (i.e. community) of the given vertex.
    
    Parameters
    ----------
    s : np.ndarray of int, shape (N,)
        Group assignment vector.
    vertex : int
        Node to focus on.
    
    Returns
    -------
    s_new : np.ndarray of int, shape (N,)
        Group assignment vector after vertex flip.
    """
    s_new = s.copy()
    s_new[vertex] = -s_new[vertex]
    return s_new

def modularity_maximization_matrix(G, B_g=None, initial_partition_matrix=None):
    """
    Refines a two–way division of the vertices in graph G to maximize modularity.
    
    Parameters
    ----------
    G : nx.Graph
        NetworkX graph.
    B_g : np.ndarray of float, shape (N, N)
        Subset of modularity matrix.
    initial_partition_matrix : np.ndarray of int, shape (N, N)
        Initial partition matrix.

    
    Returns
    -------
    final_partition_matrix : np.ndarray of int, shape (N, N)
        Final partition matrix.
    """
    N = G.number_of_nodes()
    # Initialize group vector s: if an initial partition matrix is provided, convert it; otherwise, use all ones.
    if initial_partition_matrix is None:
        s = np.ones(N, dtype=int)
    else:
        s = matrix_to_group_vector(initial_partition_matrix)

    current_modularity = compute_modularity_from_vector(B_g, s)
    improvement = True

    while improvement:
        improvement = False
        best_sweep_s = s.copy()
        best_sweep_modularity = current_modularity

        moved = set()
        temp_s = s.copy()
        temp_modularity = current_modularity
        intermediate_states = []

        # In one sweep, attempt to move each vertex at most once.
        for _ in range(N):
            best_delta = None
            best_vertex = None
            for vertex in range(N):
                if vertex in moved:
                    continue
                candidate_s = flip_vertex(temp_s, vertex)
                candidate_modularity = compute_modularity_from_vector(B_g, candidate_s)
                delta = candidate_modularity - temp_modularity
                if best_delta is None or delta > best_delta:
                    best_delta = delta
                    best_vertex = vertex

            if best_vertex is None:
                break

            # Make the best move
            temp_s = flip_vertex(temp_s, best_vertex)
            temp_modularity += best_delta
            moved.add(best_vertex)
            intermediate_states.append((temp_s.copy(), temp_modularity))

        # Roll back to the intermediate state that achieved the highest modularity.
        for state_s, state_mod in intermediate_states:
            if state_mod > best_sweep_modularity:
                best_sweep_modularity = state_mod
                best_sweep_s = state_s

        if best_sweep_modularity > current_modularity:
            s = best_sweep_s
            current_modularity = best_sweep_modularity
            improvement = True

    final_partition_matrix = group_vector_to_matrix(s)
    return final_partition_matrix

def modularity_maximization_matrix_subset(G, B_g, subset, initial_partition_matrix=None):
    """
    Refines a two–way division for a subset of vertices (given by 'subset') of graph G to maximize modularity.
    
    Parameters
    ----------
    G : nx.Graph
        NetworkX graph.
    B_g : np.ndarray of int, shape (N, N)
        Subset of modularity matrix.
    subset : Any
        Subset of vertices in graph.
    initial_partition_matrix : np.ndarray of int, shape (N, N)
        Initial partition matrix.
    
    Returns
    -------
    Any
        Computed result.
    """
    # Create the induced subgraph for the subset
    H = G.subgraph(subset).copy()

    # Run the modularity maximization on the induced subgraph H.
    refined_partition_matrix = modularity_maximization_matrix(H, B_g, initial_partition_matrix)

    # Convert the refined partition matrix back to a group vector for the subset.
    # Note: The ordering of nodes in H.nodes() may differ, so we create a mapping.
    nodes_in_subset = list(H.nodes())
    s_subset = matrix_to_group_vector(refined_partition_matrix)

    # Build a global partition dictionary. For nodes in the subset, use the refined labels;
    # for all other nodes, assign a default label (for example, 0 to indicate "not partitioned").
    global_partition = {node: 0 for node in G.nodes()}
    for idx, node in enumerate(nodes_in_subset):
        global_partition[node] = s_subset[idx]

    return refined_partition_matrix, global_partition

def spec_part_extra_bisect(A, a, m, dualW, z_curr, group, refinement=False, verbose=False):
    """
    Bisect one community using a dual-adjusted spectral split.
    
    Parameters
    ----------
    A : np.ndarray of int | float, shape (N, N)
        Adjacency / weight matrix.
    a : np.ndarray of int | float, shape (N,)
        Degree-like vector; defaults to row sums of the symmetrized adjacency.
    m : float
        Sum of degrees; sum(a).
    dualW : np.ndarray of float, shape (N, N)
        Dual matrix representing the aggregated dual arrays.
    z_curr : np.ndarray of int, shape (N, N)
        Current 2D partititon matrix.
    group : list[int]
        Subset of nodes currently being bisected.
    refinement : bool
        Enable / disable refinement operation.
    verbose : bool
        Controls the verbosity of the output. Default is False.
    
    Returns
    -------
    Any
        Computed result.
    """
    N = np.shape(A)[0]
    zii = np.zeros((N, N))

    # construct modularNty matrix
    B = (A/m - np.outer(a, a)/(m*m)) - dualW

    kdelta = np.identity(N)

    sum_Bik = [np.sum(row[group]) for row in B]

    mod_B = np.zeros_like(B)

    for i in range(N):
        for j in range(N):
            mod_B[i,j] = B[i,j] - (kdelta[i,j] * sum_Bik[i])

    if verbose != -1:
        print(group)

    B_g = mod_B[np.ix_(group, group)]

    # compute eigen-decomposition

    # For a symmetric matrix, np.linalg.eig returns real eigenvalues, but sorting is not guaranteed so we choose the eigenvector with largest eigenvalue.
    evvals, evvecs = np.linalg.eigh(B_g) # scipy.sparse.linalg.eigsh
    idx_max = np.argmax(evvals)
    evmax = evvecs[:, idx_max]

    # # PREV: Grouping by the sign of the eigenvector entries
    # # evmax[evmax < 1e-7] = -1
    # gp = np.sign(evmax)
    # print(gp)

    # # Ensure no zeros (might want to adjust if any zero occurs)
    # gp[gp == 0] = 1

    # CURR: Grouping by the sum of the eigenvector instead of the sign because the vector is changed by the duals.
    threshold = np.sum(evmax)

    gp = (evmax >= threshold).astype(int)
    gp[gp == 0] = -1

    # Transform from grouping to binary matrix zii
    zii = (np.outer(gp, gp) + 1) / 2.0

    if refinement:
        # Build local graph explicitly instead of relying on notebook global `G`.
        G_local = nx.from_numpy_array(A)
        zii, _ = modularity_maximization_matrix_subset(G_local, B_g, group, initial_partition_matrix=zii)
        if verbose != -1:
            print(zii)

    z_out = z_curr.copy()
    z_out[np.ix_(group, group)] = zii

    sub_obj_val = np.sum(mod_B * z_out)

    return gp, sub_obj_val, z_out

def full_spectral_bisection(
        A, a, m, dualW,
        refinement=False,
        verbose=False
):
    """
    Breadth First Spectral Community Structure Detection.
    
    Parameters
    ----------
    A : np.ndarray of int | float, shape (N, N)
        Adjacency / weight matrix.
    a : np.ndarray of int | float, shape (N,)
        Degree-like vector; defaults to row sums of the symmetrized adjacency.
    m : float
        Twice the total weight in the graph.
    dualW : np.ndarray of float, shape (N, N)
        Dual matrix representing the aggregated dual arrays.
    refinement : bool
        Enable / disable refinement operation.
    verbose : bool
        Controls the verbosity of the output. Default is False.
    
    Returns
    -------
    z_curr: np.ndarray of int, shape (N, N)
        Current 2D partition matrix
    sub_obj_val: float
        Subproblem objective value (reduced cost)
    """
    N = np.shape(A)[0]
    SEEN = set()
    communities = [list(range(N))]
    z = z_curr = np.ones(shape=(N, N))
    community_maps = []

    n_communities = 1
    diff = N

    while diff > 0:
        for community in map(tuple, communities):
            if community in SEEN:
                z_sol = []
            else:
                SEEN.add(community)
                gp_spec, sub_obj_val, z_sol = spec_part_extra_bisect(
                    A, a, m,
                    dualW, z_curr=z_curr, group=list(community),
                    refinement=refinement
                )
            try:
                z[np.ix_(community, community)] = z_sol[np.ix_(community, community)]
            except IndexError: # No new community is found
                 ...
        community_map, communities = group_nodes_by_community(z)
        diff = len(communities) - n_communities
        n_communities = len(communities)
        z_curr = z
        community_maps.append(community_map)
    # Return (partition_matrix, objective) to match subproblem caller expectation.
    return z_curr, sub_obj_val
