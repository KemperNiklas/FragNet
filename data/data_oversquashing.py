import os.path
import random
# from sklearn.preprocessing import LabelBinarizer
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch.nn.functional import one_hot
from torch.utils.data import random_split
from torch_geometric.data import Data
from torch_geometric.datasets import ZINC, Planetoid, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import (BaseTransform, Compose, OneHotDegree,
                                        add_positional_encoding)
from torch_geometric.utils import degree

import data.fragmentations.fragmentations as frag
from config import DATASET_ROOT
from data.fragmentations.fragmentations import FragmentData
from data.graph_to_mol import (  # I have no idea why but this import has to be in the beginning (I know that this is probably a bad sign...)
    OGB_Graph_Add_Mol_By_Smiles, ZINC_Graph_Add_Mol)
from data.lrgb import lrgb
from data.plogp import FixPlogP, LogP

"""
Adapted from https://github.com/lrnzgiusti/on-oversquashing/blob/main/data/ring_transfer.py:
Authors:
    - CWN project authors
    - On Oversquashing project authors
"""


def generate_lollipop_transfer_graph(nodes: int, target_label: List[int], frag_type: Optional[Literal["rings", "rings-paths"]] = None):
    """
    Generate a lollipop transfer graph.

    Args:
    - nodes (int): Total number of nodes in the graph.
    - target_label (list): Label of the target node.

    Returns:
    - Data: Torch geometric data structure containing graph details.
    """
    if nodes <= 1:
        raise ValueError("Minimum of two nodes required")
    # Initialize node features. The first node gets 0s, while the last gets the target label
    x = np.ones((nodes, len(target_label)))
    x[0, :] = target_label
    x[nodes - 1, :] = 0.0
    x = torch.tensor(x, dtype=torch.float32)

    edge_index = []

    # Construct a circle for the first half of the nodes,
    # where each node is connected to every other node except itself
    for i in range(nodes // 2):
        if i < nodes // 2 - 1:
            edge_index.append([i, i+1])
            edge_index.append([i+1, i])
        else:
            edge_index.append([i, 0])
            edge_index.append([0, i])

    # Construct a path (a sequence of connected nodes) for the second half of the nodes
    for i in range(nodes // 2, nodes - 1):
        edge_index.append([i, i+1])
        edge_index.append([i+1, i])

    # Connect the last node of the clique to the first node of the path
    edge_index.append([nodes // 4, nodes // 2])
    edge_index.append([nodes // 2, nodes // 4])

    # Convert the edge index list to a torch tensor
    edge_index = np.array(edge_index, dtype=np.compat.long).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Add higher level edges:
    fragments_edge_index = []
    num_fragments = 0
    if frag_type == "rings" or "rings-paths":
        for ring_node in range(nodes // 2):
            fragments_edge_index.append([ring_node, 0])
        num_fragments += 1
    if frag_type == "rings-paths":
        for path_node in range(nodes // 2, nodes):
            fragments_edge_index.append([path_node, 1])
        fragments_edge_index.append([nodes // 4, 1])
        num_fragments += 1

    fragments = torch.zeros((num_fragments, 2))
    if frag_type == "rings" or "rings-paths":
        fragments[0, 0] = 1
    if frag_type == "rings-paths":
        fragments[1, 1] = 1

    if fragments_edge_index:
        fragments_edge_index = np.array(
            fragments_edge_index, dtype=np.compat.long).T
        fragments_edge_index = torch.tensor(
            fragments_edge_index, dtype=torch.long)
    else:
        fragments_edge_index = torch.empty((2, 0), dtype=torch.long)

    higher_edge_index = []
    if frag_type == "rings-paths":
        higher_edge_index.append([0, 1])
        higher_edge_index.append([1, 0])
    if higher_edge_index:
        higher_edge_index = np.array(higher_edge_index, dtype=np.compat.long).T
        higher_edge_index = torch.tensor(higher_edge_index, dtype=torch.long)
    else:
        higher_edge_index = torch.empty((2, 0), dtype=torch.long)

    # Create a mask to indicate the target node (in this case, the first node)
    mask = torch.zeros(nodes, dtype=torch.bool)
    mask[nodes-1] = 1
    y = torch.zeros(nodes, dtype=torch.long)
    y[nodes-1] = np.argmax(target_label)

    return FragmentData(x=x, edge_index=edge_index, train_mask=mask, val_mask=mask, y=y, higher_edge_index=higher_edge_index, fragments_edge_index=fragments_edge_index, fragments=fragments)


def generate_lollipop_transfer_graph_dataset(nodes: int, classes: int = 5, samples: int = 10000, frag_type=None, **kwargs):
    """
    Generate a dataset of lollipop transfer graphs.

    Args:
    - nodes (int): Total number of nodes in each graph.
    - classes (int): Number of different classes or labels.
    - samples (int): Number of graphs in the dataset.

    Returns:
    - list[Data]: List of Torch geometric data structures.
    """
    if nodes <= 1:
        raise ValueError("Minimum of two nodes required")
    dataset = []
    samples_per_class = samples // classes
    for i in range(samples):
        label = i // samples_per_class
        target_class = np.zeros(classes)
        target_class[label] = 1.0
        graph = generate_lollipop_transfer_graph(
            nodes, target_class, frag_type)
        dataset.append(graph)
    return dataset


def generate_ring_transfer_graph(nodes, target_label, frag_type):
    """
    Generate a ring transfer graph with an option to add crosses.

    Args:
    - nodes (int): Number of nodes in the graph.
    - target_label (list): Label of the target node.
    - add_crosses (bool): Whether to add cross edges in the ring.

    Returns:
    - Data: Torch geometric data structure containing graph details.
    """
    assert nodes > 1, ValueError("Minimum of two nodes required")
    # Determine the node directly opposite to the source (node 0) in the ring
    opposite_node = nodes // 2

    # Initialise feature matrix with a uniform feature.
    # This serves as a placeholder for features of all nodes.
    x = np.ones((nodes, len(target_label)))

    # Set feature of the source node to target and the opposite node to 0
    x[0, :] = target_label
    x[opposite_node, :] = 0

    # Convert the feature matrix to a torch tensor for compatibility with Torch geometric
    x = torch.tensor(x, dtype=torch.float32)

    # List to store edge connections in the graph
    edge_index = []
    for i in range(nodes-1):
        # Regular connections that make the ring
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])

    # Close the ring by connecting the last and the first nodes
    edge_index.append([0, nodes - 1])
    edge_index.append([nodes - 1, 0])

    # Convert edge list to a torch tensor
    edge_index = np.array(edge_index, dtype=np.compat.long).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Add higher level edges:
    fragments_edge_index = []
    num_fragments = 0
    if frag_type == "rings" or "rings-paths":
        for ring_node in range(nodes):
            fragments_edge_index.append([ring_node, 0])
        num_fragments += 1

    fragments = torch.zeros((num_fragments, 2))
    if frag_type == "rings" or "rings-paths":
        fragments[0, 0] = 1

    if fragments_edge_index:
        fragments_edge_index = np.array(
            fragments_edge_index, dtype=np.compat.long).T
        fragments_edge_index = torch.tensor(
            fragments_edge_index, dtype=torch.long)
    else:
        fragments_edge_index = torch.empty((2, 0), dtype=torch.long)

    higher_edge_index = torch.empty((2, 0), dtype=torch.long)

    # Create a mask to identify the target node in the graph. Only the source node (index 0) is marked true.
    mask = torch.zeros(nodes, dtype=torch.bool)
    mask[opposite_node] = 1
    y = torch.zeros(nodes, dtype=torch.long)
    y[opposite_node] = np.argmax(target_label)

    # Return the graph with nodes, edges, mask and the label
    return FragmentData(x=x, edge_index=edge_index, train_mask=mask, val_mask=mask, y=y, higher_edge_index=higher_edge_index, fragments_edge_index=fragments_edge_index, fragments=fragments)


def generate_ring_transfer_graph_dataset(nodes: int, classes: int = 5, samples: int = 10000, frag_type=None, **kwargs):
    """
    Generate a dataset of ring transfer graphs.

    Args:
    - nodes (int): Number of nodes in each graph.
    - add_crosses (bool): Whether to add cross edges in the ring.
    - classes (int): Number of different classes or labels.
    - samples (int): Number of graphs in the dataset.

    Returns:
    - list[Data]: List of Torch geometric data structures.
    """
    if nodes <= 1:
        raise ValueError("Minimum of two nodes required")
    dataset = []
    samples_per_class = samples // classes
    for i in range(samples):
        label = i // samples_per_class
        target_class = np.zeros(classes)
        target_class[label] = 1.0
        graph = generate_ring_transfer_graph(nodes, target_class, frag_type)
        dataset.append(graph)
    return dataset


def generate_handcuffs_transfer_graph(target_node: str, target_label, frag_type):
    """
    Generate a handcuffs transfer graph.

    Args:
    - target_node (str): Name of the target node.
    - target_label (list): Label of the target node.

    Returns:
    - Data: Torch geometric data structure containing graph details.
    """
    len_first_cylce = 5
    len_path = 3
    len_second_cycle = 6
    name_to_idx = {f"L{i}": i for i in range(len_first_cylce)}  # first cycle
    name_to_idx.update(
        {f"P{i}": i + len_first_cylce for i in range(len_path)})  # path
    name_to_idx.update({f"R{i}": i + len_first_cylce +
                       len_path for i in range(len_second_cycle)})  # second cycle

    num_nodes = len(name_to_idx)
    start_idx = name_to_idx["L2"]
    target_idx = name_to_idx[target_node]

    # Initialise feature matrix with a uniform feature.
    # This serves as a placeholder for features of all nodes.
    x = np.ones((num_nodes, len(target_label)))
    # x = np.zeros((num_nodes, len(target_label)))

    # Set feature of the source node to target and the opposite node to 0
    x[start_idx, :] = target_label
    x[target_idx, :] = 0

    # Convert the feature matrix to a torch tensor for compatibility with Torch geometric
    x = torch.tensor(x, dtype=torch.float32)

    # List to store edge connections in the graph
    edge_index = []
    for i in range(len_first_cylce):
        # Regular connections that make the ring
        if i < len_first_cylce - 1:
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])
        else:
            edge_index.append([i, 0])
            edge_index.append([0, i])

    for i in range(len_first_cylce, len_first_cylce + len_path - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])

    edge_index.append([name_to_idx["L1"], name_to_idx["P0"]])
    edge_index.append([name_to_idx["P0"], name_to_idx["L1"]])
    edge_index.append([name_to_idx["R1"], name_to_idx[f"P{len_path - 1}"]])
    edge_index.append([name_to_idx[f"P{len_path - 1}"], name_to_idx["R1"]])

    for i in range(len_first_cylce + len_path, num_nodes - 1):
        if i < num_nodes - 1:
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])
        else:
            edge_index.append([i, len_first_cylce + len_path])
            edge_index.append([len_first_cylce + len_path, i])

    # Convert edge list to a torch tensor
    edge_index = np.array(edge_index, dtype=np.compat.long).T
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    # Add higher level edges:
    fragments_edge_index = []
    num_fragments = 0
    if frag_type == "rings" or "rings-paths":
        for node in range(len_first_cylce):
            node_id = name_to_idx[f"L{node}"]
            fragments_edge_index.append([node_id, 0])
        for node in range(len_second_cycle):
            node_id = name_to_idx[f"R{node}"]
            fragments_edge_index.append([node_id, 1])
        num_fragments += 2

    if frag_type == "rings-paths":
        for node in range(len_path):
            node_id = name_to_idx[f"P{node}"]
            fragments_edge_index.append([node_id, 2])
        fragments_edge_index.append([name_to_idx["R1"], 2])
        fragments_edge_index.append([name_to_idx["L1"], 2])
        num_fragments += 1

    fragments = torch.zeros((num_fragments, 2))
    if frag_type == "rings" or "rings-paths":
        fragments[0, 0] = 1
        fragments[1, 0] = 1
    if frag_type == "rings-paths":
        fragments[2, 1] = 1

    if fragments_edge_index:
        fragments_edge_index = np.array(
            fragments_edge_index, dtype=np.compat.long).T
        fragments_edge_index = torch.tensor(
            fragments_edge_index, dtype=torch.long)
    else:
        fragments_edge_index = torch.empty((2, 0), dtype=torch.long)

    higher_edge_index = []
    if frag_type == "rings-paths":
        higher_edge_index.append([0, 1])
        higher_edge_index.append([1, 0])
        higher_edge_index.append([1, 2])
        higher_edge_index.append([2, 1])
    if higher_edge_index:
        higher_edge_index = np.array(higher_edge_index, dtype=np.compat.long).T
        higher_edge_index = torch.tensor(higher_edge_index, dtype=torch.long)
    else:
        higher_edge_index = torch.empty((2, 0), dtype=torch.long)

    # Create a mask to identify the target node in the graph. Only the source node (index 0) is marked true.
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[target_idx] = 1
    y = torch.zeros(num_nodes, dtype=torch.long)
    y[target_idx] = np.argmax(target_label)

    return FragmentData(x=x, edge_index=edge_index, train_mask=mask, val_mask=mask, y=y, higher_edge_index=higher_edge_index, fragments_edge_index=fragments_edge_index, fragments=fragments)


def generate_handcuffs_transfer_graph_dataset(node: str, classes: int = 5, samples: int = 10000, frag_type=None, **kwargs):
    """
    Generate a dataset of ring transfer graphs.

    Args:
    - nodes (int): Name of the node to be the target node.
    - classes (int): Number of different classes or labels.
    - samples (int): Number of graphs in the dataset.
    - frag_type (str): Type of fragments to add to the graph.

    Returns:
    - list[Data]: List of Torch geometric data structures.
    """

    dataset = []
    samples_per_class = samples // classes
    for i in range(samples):
        label = i // samples_per_class
        target_class = np.zeros(classes)
        target_class[label] = 1.0
        graph = generate_handcuffs_transfer_graph(
            node, target_class, frag_type)
        dataset.append(graph)
    return dataset


def load_fragmentation(dataset,
                       loader_params,
                       dataset_seed=None,
                       dataset_params={},
                       frag_type=None):

    if dataset == "lollipop":
        train_data = generate_lollipop_transfer_graph_dataset(
            dataset_params["num_nodes"], dataset_params["num_classes"], dataset_params["num_samples"], frag_type=frag_type)
        val_data = generate_lollipop_transfer_graph_dataset(
            dataset_params["num_nodes"], dataset_params["num_classes"], dataset_params["num_val_samples"], frag_type=frag_type)
        test_data = []
    elif dataset == "ring":
        train_data = generate_ring_transfer_graph_dataset(
            dataset_params["num_nodes"], dataset_params["num_classes"], dataset_params["num_samples"], frag_type=frag_type)
        val_data = generate_ring_transfer_graph_dataset(
            dataset_params["num_nodes"], dataset_params["num_classes"], dataset_params["num_val_samples"], frag_type=frag_type)
        test_data = []
    elif dataset == "handcuffs":
        train_data = generate_handcuffs_transfer_graph_dataset(
            dataset_params["target_node"], dataset_params["num_classes"], dataset_params["num_samples"], frag_type=frag_type)
        val_data = generate_handcuffs_transfer_graph_dataset(
            dataset_params["target_node"], dataset_params["num_classes"], dataset_params["num_val_samples"], frag_type=frag_type)
        test_data = []

    follow_batch = ["x", "fragments"]

    if not dataset_seed:
        train_loader = DataLoader(train_data,
                                  batch_size=loader_params["batch_size"],
                                  num_workers=loader_params["num_workers"],
                                  follow_batch=follow_batch,
                                  shuffle=True)
    else:
        # set seperate seed for dataloaderr
        g = torch.Generator()
        g.manual_seed(dataset_seed)
        train_loader = DataLoader(train_data,
                                  batch_size=loader_params["batch_size"],
                                  num_workers=loader_params["num_workers"],
                                  follow_batch=follow_batch,
                                  shuffle=True,
                                  generator=g)

    val_batch_size = loader_params[
        "val_batch_size"] if "val_batch_size" in loader_params else loader_params[
            "batch_size"]
    val_loader = DataLoader(val_data,
                            batch_size=val_batch_size,
                            num_workers=loader_params["num_workers"],
                            follow_batch=follow_batch)
    test_loader = DataLoader(test_data,
                             batch_size=val_batch_size,
                             num_workers=loader_params["num_workers"],
                             follow_batch=follow_batch)

    num_features = dataset_params["num_classes"]
    num_classes = dataset_params["num_classes"]
    return train_loader, val_loader, test_loader, num_features, num_classes
