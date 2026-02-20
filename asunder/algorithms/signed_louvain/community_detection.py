#!/usr/bin/env python
# coding: utf-8
# Source: https://github.com/lejohnyjohn/signed-louvain/blob/main/community_detection.py
# Description: Taken from repo on July 18, 2025
# In[ ]:

"""Signed Louvain community detection implementation vendored for Asunder."""


import numbers
import time
from copy import deepcopy

import networkx as nx
import numpy as np

# In[ ]:


def check_random_state(seed):
    """ 
    Turns seed into a np.random.RandomState instance.
    ----------
    input
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState"
                     " instance" % seed)


# In[ ]:


class Wrapper(object):
    """Mixin with partition update helpers used by signed Louvain status objects."""

    def __init__(self):
        pass
    
    def randomize(self, items):
        """ Returns a list containing a random permutation of items. """
        randomized_items = items #list(items)
        self.random_state.shuffle(randomized_items)
        return randomized_items

    def remove(self, node, com, neigh_communities): 
        """ Removes node `node` from community `com`, and updates status's metrics. """
        weights = [neigh_communities[idx].get(com, 0.) for idx in range(self.nb_layers)]
        for idx in range(self.nb_layers):
            self.degrees[idx][com] = self.degrees[idx].get(com, 0.) - self.gdegrees[idx].get(node, 0.)
            self.internals[idx][com] = float(self.internals[idx].get(com, 0.)                                              - weights[idx]                                              - self.loops[idx].get(node, 0.))
        self.node2com[node] = -1
        
    def insert(self, node, com, weights):
        """ Inserts node in community `com` and updates status. """
        for idx in range(self.nb_layers):
            self.degrees[idx][com] = self.degrees[idx].get(com, 0.) + self.gdegrees[idx].get(node, 0.)
            self.internals[idx][com] = float(self.internals[idx].get(com, 0.)                                              + weights[idx]                                              + self.loops[idx].get(node, 0.))
        self.node2com[node] = com
        
    def renumber(self):
        """ Renumbers the labels of the communities from 0 to n. """
        values = set(self.node2com.values())
        target = set(range(len(values)))
        if values == target:
            ret = self.node2com
        else:
            # Add the values that won't be renumbered
            renumbering = dict(zip(target.intersection(values),
                                   target.intersection(values)))
            # Add the values that will be renumbered
            renumbering.update(dict(zip(values.difference(target),
                                        target.difference(values)))
                              )
            ret = {k: renumbering[v] for k, v in self.node2com.items()}
        return ret
    
    def compute_neigh_communities(self, node):
        """
        Computes the communities in the node's neighborhood 
        based on partition status.node2com
        """
        weight_key = self.weight
        weights = {idx: {} for idx in range(self.nb_layers)}
        for idx, layer in enumerate(self.layers):
            for neighbor, data in layer[node].items():
                if neighbor != node:
                    edge_weight = data.get(weight_key, 1)
                    neighborcom = self.node2com[neighbor]
                    weights[idx][neighborcom] = weights[idx].get(neighborcom, 0) + edge_weight
        return weights
    
    def find_k_hop_communities(self, node, k=2):
        """
        Finds up-to-k-hop communities of node 
        based on partition status.node2com
        """
        if k < 0:
            k_hop_communities = self.randomize(list(self.node2com.values()))
        elif k > 0:
            if node in self.k_hop_graph:
                coms = []
                for key in self.k_hop_graph[node]:
                    coms += [self.node2com[cur_node]                              for cur_node in self.randomize(self.k_hop_graph[node][key])]
                k_hop_communities = list(set(coms))
        return k_hop_communities
    
    def format_neigh_communities(self, neigh_communities):
        """
        Gathers all communities candidates
        to move a node based on partition given by `status`.
        """
        new_neigh_communities = {}
        for idx, layer in enumerate(self.layers):
            for comm, data in neigh_communities[idx].items():
                if comm not in new_neigh_communities:
                    values = {idy: 0. for idy in range(idx)}
                    values[idx] = data
                    for idy in range(idx + 1, self.nb_layers):
                        if comm in neigh_communities[idy]:
                            values[idy] = neigh_communities[idy][comm]
                        else:
                            values[idy] = 0.
                    new_neigh_communities[comm] = values 
        return new_neigh_communities
    
    def compute_remove_cost(self, neigh_communities, node, com_node, degc_totws):
        """
        Computes cost of removing node from a given community `com_node`
        based on partition given by `status`.
        """
        total_cost = 0.
        for idx in range(self.nb_layers):
            value = self.resolutions[idx] 
            value *= (self.degrees[idx].get(com_node, 0.) - self.gdegrees[idx].get(node, 0.))
            value *= degc_totws[idx] 
            value -= neigh_communities[idx].get(com_node, 0)
            total_cost += self.layer_weights[idx] * value
        return total_cost
    
    def compute_insert_cost(self, com, degree_in_coms, degc_totws):
        """
        Computes cost of inserting node in a given community `com`
        based on partition given by `status`.
        """
        total_cost = 0
        for idx in range(self.nb_layers):
            value = degree_in_coms[idx]
            value -= self.resolutions[idx] * self.degrees[idx].get(com, 0.) * degc_totws[idx]
            total_cost += self.layer_weights[idx] * value
        return total_cost
    
    def modularity(self):
        """ 
        Computes the modularity of the partition of the graph
        using the stored status.
        ----
        input 
        status: Status object
        """
        result = 0.
        for idx, layer in enumerate(self.layers):
            links = float(self.total_weight[idx])
            for com in set(self.node2com.values()):
                in_degree = self.internals[idx].get(com, 0.)
                degree = self.degrees[idx].get(com, 0.)
                value = in_degree * self.resolutions[idx] / links -  ((degree / (2. * links)) ** 2)
                value *= self.layer_weights[idx]
                result += value 
        return result

    def one_level(self, pass_max, epsilon, k, lowest_level=True):
        """ Computes one level of communities. """
        modified = True
        nb_pass_done = 0
        cur_mod = self.modularity()
        new_mod = cur_mod
        while modified and nb_pass_done < pass_max:
            cur_mod = new_mod
            modified = False
            nb_pass_done += 1

            for node in self.randomize(self.nodes):
                com_node = self.node2com[node]
                degc_totws = [self.gdegrees[idx].get(node, 0.) / (self.total_weight[idx] * 2.)                               for idx in range(self.nb_layers)]
                neigh_communities = self.compute_neigh_communities(node)
                remove_cost = self.compute_remove_cost(neigh_communities, node, com_node, degc_totws)
                self.remove(node, com_node, neigh_communities)
                
                if lowest_level:
                    if self.consider_empty_community:
                        best_com = -1
                        best_increase = remove_cost
                    else:
                        best_com = com_node
                        best_increase = 0
                else:
                    best_com = com_node
                    best_increase = 0
                neigh_communities = self.format_neigh_communities(neigh_communities)
                other_candidates = [com_node] + self.find_k_hop_communities(node, k=k)
                candidate_communities = list(set(list(neigh_communities.keys()) + other_candidates))
                base = {idx: 0. for idx in range(self.nb_layers)}
                for com in candidate_communities:
                    if com in neigh_communities:
                        incr = remove_cost + self.compute_insert_cost(com, neigh_communities[com], degc_totws)
                    else:
                        incr = remove_cost + self.compute_insert_cost(com, base, degc_totws)
                    if incr > best_increase:
                        best_increase = incr
                        best_com = com
                if best_com == -1:
                    self.insert(node, max(list(self.node2com.values())), base)
                elif best_com in neigh_communities:
                    self.insert(node, best_com, neigh_communities[best_com]) 
                else:
                    self.insert(node, best_com, base)
                if best_com != com_node:
                    modified = True
            new_mod = self.modularity()
            if new_mod - cur_mod < epsilon:
                break
    


# In[ ]:


class Status(Wrapper):
    """ 
    Stores and acts on the current (multilayer) graph, as well as
    the relevant node and community metrics (degrees etc.).
    """
    
    def __init__(self, 
                 layers, 
                 k, 
                 resolutions, 
                 layer_weights, 
                 masks, 
                 consider_empty_community,
                 random_state=None, 
                 weight_label='weight'):
        """ 
        input
        layers : list
            list of nx.Graph objects representing each layer
        resolutions : list
            list of resolution parameters (int/float) associated to each layer
        layer_weights : list
            list of weights (int/float) associated to each layer
        masks : list of booleans
            list of masks (for multiple-hop communities) associated to each layer
        weight_label : str
            the name of the edge weight label in the graph data
        """
        self.start_time = time.time()
        self.times_to_build_k_hop_graph = []
        self.k = k
        self.consider_empty_community = consider_empty_community
        self._retain_layers_with_edges(layers, resolutions, layer_weights, masks)
        self.nodes = sorted(list(self.layers[0].nodes()))
        self.nb_layers = len(self.layers)
        self.weight = weight_label
        self.random_state = random_state
        
        self.node2com = {}
        self.total_weight = {idx: 0. for idx in range(self.nb_layers)} # total weight (in a layer)
        self.degrees = {idx: dict([]) for idx in range(self.nb_layers)} # sum of degrees in a community (in a layer)
        self.gdegrees = {idx: dict([]) for idx in range(self.nb_layers)} # degree of a node (in a layer)
        self.internals = {idx: dict([]) for idx in range(self.nb_layers)} # sum of weights within a community (in a layer)
        self.loops = {idx: dict([]) for idx in range(self.nb_layers)} # weight of self-loops (in a layer)
    
    def _retain_layers_with_edges(self, layers, resolutions, layer_weights, masks):
        """ 
        Filters out layers with no edges.
        ----------
        input: same spec as in self.__init__()
        """
        self.layers = []
        self.resolutions = []
        self.layer_weights = []
        self.masks = []
        for idx, layer in enumerate(layers):
            if layer.number_of_edges() > 0:
                self.layers += [layer]
                self.resolutions += [resolutions[idx]]
                self.layer_weights += [layer_weights[idx]]
                self.masks += [masks[idx]]
                
    def update_layers(self, layers):
        """ 
        Updates layers when the graph is made coarser.
        ----------
        input
        layers : list
            list of nx.Graph objects representing each layer
        """
        self.layers = layers
        self.nodes = sorted(list(self.layers[0].nodes()))
    
    def _compute_total_weights(self):
        """ Computes total weight for each layer. """
        return {idx: layer.size(self.weight) for idx, layer in enumerate(self.layers)}
    
    def _compute_degrees(self, node):
        """ Computes node's degree in each layer. """
        return [float(layer.degree(node, weight=self.weight)) for layer in self.layers]
    
    def _get_self_loop_data(self, node):
        """ Computes node's self-loop degree in each layer. """
        return [float(layer.get_edge_data(node, node, default={self.weight: 0})[self.weight])                 for layer in self.layers]
    
    def _compute_inc(self, node, com, part):
        """ Computes sum of the weights between node `node` and 
            community `com` from partition `part`. """
        incs = []
        for layer in self.layers:
            inc = 0.
            for neighbor, data in layer[node].items():
                if part[neighbor] == com:
                    edge_weight = data[self.weight]
                    if neighbor == node:
                        inc += float(edge_weight)
                    else:
                        inc += float(edge_weight) / 2.
            incs += [inc]
        return incs
    
    def _build_k_hop_graph(self):
        update_k = False
        if self.k < 0:
            self.k = 1
            update_k = True
        k_hop_graph = {node: {distance: [] for distance in range(1, self.k + 1)} for node in self.nodes}
        for idx, layer in enumerate(self.layers):
            if self.masks[idx]:
                temp_k = self.k
            else:
                temp_k = 1
            path = dict(nx.all_pairs_shortest_path_length(layer, cutoff=temp_k))
            for node in path:
                node_path = path[node]
                for cur_node in node_path:
                    distance = node_path[cur_node]
                    if distance > 0:
                        k_hop_graph[node][distance] += [cur_node]
        for node in k_hop_graph:
            for distance in range(1, self.k + 1):
                k_hop_graph[node][distance] = list(set(k_hop_graph[node][distance]))      
        if update_k:
            self.k = -1
        self.k_hop_graph = k_hop_graph
                                    
    def init(self, part=None):
        """ 
        Initialises the status of a graph with every node in one community, 
        unless a specific partition `part` is provided.
        """
        start_time = time.time()
        self._build_k_hop_graph()
        self.times_to_build_k_hop_graph += [time.time() - start_time]
        
        count = 0
        self.node2com = {} 
        self.total_weight = {idx: 0. for idx in range(self.nb_layers)}
        self.degrees = {idx: dict([]) for idx in range(self.nb_layers)}
        self.gdegrees = {idx: dict([]) for idx in range(self.nb_layers)}
        self.internals = {idx: dict([]) for idx in range(self.nb_layers)}
        
        self.total_weight = self._compute_total_weights() 
        if part is None:
            for node in self.nodes:
                self.node2com[node] = count
                degs = self._compute_degrees(node)
                for idx in range(self.nb_layers):
                    self.degrees[idx][count] = degs[idx]
                    self.gdegrees[idx][node] = degs[idx]
                loops = self._get_self_loop_data(node)
                for idx in range(self.nb_layers):
                    self.loops[idx][node] = loops[idx]
                    self.internals[idx][count] = self.loops[idx][node]
                count += 1
        else:
            for node in self.nodes:
                com = part[node]
                self.node2com[node] = com
                degs = self._compute_degrees(node)
                for idx in range(self.nb_layers):
                    self.degrees[idx][com] = self.degrees[idx].get(com, 0) + degs[idx]
                    self.gdegrees[idx][node] = degs[idx]
                loops = self._get_self_loop_data(node)
                incs = self._compute_inc(node, com, part)
                for idx in range(self.nb_layers):
                    self.loops[idx][node] = loops[idx]
                    self.internals[idx][com] = self.internals[idx].get(com, 0) + incs[idx]
    
    def induced_graph(self, partition):
        """ 
        Produces graph where nodes are the communities
        There is a link of weight w between communities if the sum of the weights
        of the links between their elements is w.
        ----------
        input
        partition : dict
           a dictionary where keys are graph nodes and values the part the node
           belongs to
        status : Status object
            the initial graph
    
        returns
        coarse_layers : list
            list of coarse-grained layers
        """
        weight_label = self.weight
        coarse_layers = []
        for layer in self.layers:
            coarse_layer = nx.Graph()
            coarse_layer.add_nodes_from(partition.values())
            for node1, node2, datas in layer.edges(data=True):
                edge_weight = datas[weight_label]
                com1 = partition[node1]
                com2 = partition[node2]
                w_prec = coarse_layer.get_edge_data(com1, com2, default={weight_label: 0})[weight_label]
                coarse_layer.add_edge(com1, com2, **{weight_label: w_prec + edge_weight})
            coarse_layers += [coarse_layer]
    
        return coarse_layers

    def describe(self):
        print("node2com : " + str(self.node2com))
        print("degrees : " + str(self.degrees))
        print("internals : " + str(self.internals))
        print("total_weight : " + str(self.total_weight))


# In[ ]:


def generate_dendrogram(layers=[nx.Graph(), nx.Graph()],
                        k=2,
                        layer_weights=[1., -1.],
                        resolutions=[1., 1.],
                        masks=[False, True],
                        initial_membership=None,
                        weight='weight',
                        random_state=None, 
                        pass_max=10,
                        epsilon=1e-7, 
                        consider_empty_community=False,
                        silent=True):
    """
    Find communities in a (multiplex) graph and returns the associated dendrogram.
    ----------
    input
    layers : list
        list of nx.Graph objects representing each layer
    layer_weights : list
        list of weights (int/float) associated to each layer
    resolutions : list
        list of resolution parameters (int/float) associated to each layer
    masks : list of booleans
        list of masks (for k-hop communities) associated to each layer
    k : int
        exploration depth i.e. number of hops of neighbours to scan 
        k=1 is classic Louvain and scans the current node's neighbouring communities only
        k=2 scans communities of the direct and second-hop neighbours of the current node
        k<0 scans all communities in the graph 
    initial_membership: dict
        initial partition of the nodes
    weight : str
        the name of the edge weight label in the graph data
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    pass_max : int
        number of iterations across all nodes for commnity updates at each level
    epsilon : float
        stop modularity updates if value improvements are smaller than epsilon
    
    returns 
    dendrogram : list of dictionaries
        a list of partitions, ie dictionnaries where keys of the i+1 are the
        values of the i. and where keys of the first are the nodes of graph
    ------
    TypeError
        If graph is empty
        If the graph is not a nx.Graph (undirected)
        If layers, resolutions, layer_weights and masks do not have the same length
        If layers do not have the same nodes
        If the exploration depth `k` is not a non-zero integer
    """
    if len(layers) < 1:
        raise TypeError("No graph in input.")
        
    if len(layers) != len(resolutions):
        raise TypeError("Graph list and resolution list must be of same size.")
    if len(layers) != len(layer_weights):
        raise TypeError("Graph list and resolution list must be of same size.")
    if len(layers) != len(masks):
        raise TypeError("Graph list and mask list must be of same size.")
        
    if not isinstance(k, int) or k == 0:
        raise TypeError("The exploration depth `k` must be a non-zero integer.")
    
    nodes = sorted(list(layers[0].nodes()))
    for layer in layers:
        if sorted(list(layer.nodes())) != nodes:
            raise TypeError("All layers must have the same nodes.")
            
    for layer in layers:
        if layer.is_directed():
            raise TypeError("Please use non-directed graphs only.")

    random_state = check_random_state(random_state)
    
    # When there is no edge in the multiplex network,
    # The best partition is everyone in its community
    flag = True
    for layer in layers:
        if layer.number_of_edges() > 0:
            flag = False
    if flag:
        part = dict([])
        for i, node in enumerate(layers[0].nodes()):
            part[node] = i
        return [part]
    
    cur_layers = layers.copy()
    status = Status(layers, 
                    k, 
                    resolutions, 
                    layer_weights, 
                    masks, 
                    consider_empty_community, 
                    random_state=random_state, 
                    weight_label=weight)
    status.init(part=initial_membership)
    status_list = list()
    #print('--- first level ---')
    #print('initial membership:', initial_membership)
    status.one_level(pass_max, epsilon, k, lowest_level=True)
    partition = status.renumber()
    #print('partition:', partition)
    mod = status.modularity()
    status_list.append((partition, status))
    cur_layers = status.induced_graph(partition)
    status.update_layers(cur_layers)
    status.init(part=None)
   
    while True:
        #print('--- next level ---')
        status.one_level(pass_max, epsilon, k, lowest_level=False)
        new_mod = status.modularity()
        partition = status.renumber()
        #print('partition:', partition)
        status_list.append((partition, status))
        if new_mod - mod < epsilon:
            break
        mod = new_mod
        cur_layers = status.induced_graph(partition)
        status.update_layers(cur_layers)
        status.init(part=None)
    if not silent:
        print('computing time:', time.time() - status.start_time)
        print('computing times to build k-graph:', status.times_to_build_k_hop_graph)
    return status_list[:]


# In[ ]:


def partition_at_level(dendrogram, level):
    """
    Returns the partition of the nodes at the given level
    
    A dendrogram is a tree and each level is a partition of the graph nodes.
    Level 0 is the first partition, which contains the smallest communities,
    and the best is len(dendrogram) - 1.
    
    The higher the level is, the bigger are the communities.
    ----------
    input
    dendrogram : list of dict
       a list of partitions, ie dictionnaries where keys of the i+1 are the
       values of the i.
    level : int
       the level which belongs to [0..len(dendrogram)-1]
    
    returns
    partition : dictionnary
       A dictionary where keys are the nodes and the values are the set it
       belongs to
    ------
    KeyError
       If the dendrogram is not well formed or the level is too high
    """
    partition, status = deepcopy(dendrogram[0])
    for index in range(1, level + 1):
        temp_dendrogram, temp_status = dendrogram[index]
        for node, community in partition.items():
            partition[node] = temp_dendrogram[community]
        status = temp_status
    return partition, status


# In[ ]:


def best_partition(layers=[nx.Graph(), nx.Graph()],
                   k=2,
                   layer_weights=[1., -1.],
                   resolutions=[1., 1.],
                   masks=[False, True],
                   initial_membership=None,
                   weight='weight',
                   random_state=None, 
                   pass_max=10,
                   epsilon=1e-7, 
                   return_dendogram=False, 
                   consider_empty_community=False,
                   silent=True):
    """ 
    Computes the partition of the graph nodes which tries to maximise the modularity
    using a Louvain-inspired heuristics.  
    This is the partition of highest modularity, i.e. the highest partition
    of the dendrogram generated by the Louvain-inspired algorithm.
    ----------
    input
    layers : list
        list of nx.Graph objects representing each layer
    layer_weights : list
        list of weights (int/float) associated to each layer
    resolutions : list
        list of resolution parameters (int/float) associated to each layer
    masks : list of booleans
        list of masks (for k-hop communities) associated to each layer
    k : int
        exploration depth i.e. number of hops of neighbours to scan 
        k=1 is classic Louvain and scans the current node's neighbouring communities only
        k=2 scans communities of the direct and second-hop neighbours of the current node
        k<0 scans all communities in the graph 
    initial_membership: dict
        initial partition of the nodes
    weight : str
        the name of the edge weight label in the graph data
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    pass_max : int
        number of iterations across all nodes for commnity updates at each level
    epsilon : float
        stop modularity updates if value improvements are smaller than epsilon
    return_dendogram : bool
        if False, the function returns the last level of the dendogram (the coarse communities) 
        if True, the function returns all levels of the dendogram
        
    returns
    partition : dict (if return_dendogram=True) or list (if return_dendogram=False)
       if return_dendogram=False: the coarse partition
       if return_dendogram=True: the partitions at all levels of the dendogram
       At any level, the communities are numbered from 0 to number of communities - 1
    ------
    TypeError
        If graph is empty
        If the graph is not a nx.Graph (undirected)
        If layers, resolutions and layer_weights do not have the same length
        If layers do not have the same nodes
    --------
    See also
    generate_dendrogram : to obtain all the decompositions levels
    """
    dendo = generate_dendrogram(layers=layers,
                                layer_weights=layer_weights,
                                resolutions=resolutions,
                                masks=masks,
                                k=k,
                                initial_membership=initial_membership,
                                weight=weight,
                                random_state=random_state, 
                                pass_max=pass_max,
                                epsilon=epsilon, 
                                consider_empty_community=consider_empty_community,
                                silent=silent)
    if return_dendogram:
        return dendo
    else:
        return partition_at_level(dendo, len(dendo) - 1)

