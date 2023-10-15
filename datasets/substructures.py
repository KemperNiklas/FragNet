import graph_tool as gt
import graph_tool.generation as gen
import graph_tool.topology as top
import networkx as nx

import torch
from itertools import chain

from collections import defaultdict
from math import comb

def get_cliques(edge_index: torch.Tensor, max_k: int, min_k: int = 3):
    """
    Compute all cliques of given lengths.

    Parameters
    ----------
    edge_index
        Pytorch-geometric style edge index.
    max_k
        Maximum number of nodes in ring
    min_k, optional
        Minimum number of nodes in ring, by default 3

    Returns
    -------
        List of sets of all cliques with length >= min_k and length <= max_k. Permutations of cliques are removed.
    """
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()

    edge_list = edge_index.T
    graph_gt = gt.Graph(directed=False)
    graph_gt.add_edge_list(edge_list)
    gen.remove_self_loops(graph_gt)
    gen.remove_parallel_edges(graph_gt)
    cliques = set()
    sorted_cliques = set()
    for k in range(min_k, max_k+1):
        pattern = nx.complete_graph(k)
        pattern_edge_list = list(pattern.edges)
        pattern_gt = gt.Graph(directed=False)
        pattern_gt.add_edge_list(pattern_edge_list)
        sub_isos = top.subgraph_isomorphism(pattern_gt, graph_gt, induced=True, subgraph=True,
                                           generator=True)
        sub_iso_sets = map(lambda isomorphism: tuple(isomorphism.a), sub_isos)
        for iso in sub_iso_sets:
            if tuple(sorted(iso)) not in sorted_cliques:
                cliques.add(iso)
                sorted_cliques.add(tuple(sorted(iso)))
    cliques = list(cliques)
    return cliques

def get_max_cliques(edge_index: torch.Tensor):
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()
    edge_list = edge_index.T
    graph_gt = gt.Graph(directed=False)
    graph_gt.add_edge_list(edge_list)
    gen.remove_self_loops(graph_gt)
    gen.remove_parallel_edges(graph_gt)
    return top.max_cliques(graph_gt)

def get_clique_counts(edge_index: torch.Tensor, max_clique):
    clique_counts = [0 for i in range(3, max_clique +1)]
    for clique in get_max_cliques(edge_index):
        if len(clique) < 3:
            continue
        elif len(clique) > max_clique:
            clique_counts[max_clique - 3] += comb(len(clique), max_clique)
        else:
            clique_counts[len(clique) - 3] += 1
    return clique_counts


def get_rings(edge_index: torch.Tensor, max_k: int, min_k: int = 3):
    """
    Compute all rings of given lengths.

    Parameters
    ----------
    edge_index
        Pytorch-geometric style edge index.
    max_k
        Maximum number of nodes in ring
    min_k, optional
        Minimum number of nodes in ring, by default 3

    Returns
    -------
        List of sets of all rings with length >= min_k and length <= max_k. Permutations of rings are removed.
    """
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()

    edge_list = edge_index.T
    graph_gt = gt.Graph(directed=False)
    graph_gt.add_edge_list(edge_list)
    gen.remove_self_loops(graph_gt)
    gen.remove_parallel_edges(graph_gt)
    rings = set()
    sorted_rings = set()
    for k in range(min_k, max_k+1):
        pattern = nx.cycle_graph(k)
        pattern_edge_list = list(pattern.edges)
        pattern_gt = gt.Graph(directed=False)
        pattern_gt.add_edge_list(pattern_edge_list)
        sub_isos = top.subgraph_isomorphism(pattern_gt, graph_gt, induced=True, subgraph=True,
                                           generator=True)
        sub_iso_sets = map(lambda isomorphism: tuple(isomorphism.a), sub_isos)
        for iso in sub_iso_sets:
            if tuple(sorted(iso)) not in sorted_rings:
                rings.add(iso)
                sorted_rings.add(tuple(sorted(iso)))
    rings = list(rings)
    return rings

def get_node_counts(substructure, num_nodes: int) -> torch.Tensor:
    """Compute counts of appearances in the substructures for each node.

    Parameters
    ----------
    substructure
        List of substructure tuples with node ids.
    num_nodes
        Number of nodes in the dataset.

    Returns
    -------
        Tensor of shape (num_nodes,) with the number of appearances in the substructures for each node.
    """
    return torch.tensor([list(chain(*substructure)).count(i) for i in range(num_nodes)], dtype = torch.float)

def get_substructure_edge_index(substructure):
    """Compute node-to-substructure edge index.

    Parameters
    ----------
    substructure
        List of substructure tuples with node ids.

    Returns
    -------
        Pytorch-geometric style edge index from nodes to substructures in which the nodes are part of.
    """
    if not substructure:
        return torch.empty(size = (2,0), dtype = torch.long)
    return torch.tensor([[node_id, sub_id]  for sub_id, sub in enumerate(substructure) for node_id in sub], dtype = torch.long).t().contiguous()