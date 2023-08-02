from typing import Dict, List, Optional, Tuple
from torch_geometric.datasets import TUDataset, Planetoid
from torch_geometric.loader import DataLoader
import datasets.substructures as sub
import torch

TU_DATASETS = ["MUTAG", "ENZYMES", "PROTEINS", "COLLAB", "IMDB_BINARY", "REDDIT-BINARY"]
PLANETOID_DATASETS = ["Cora", "CiteSeer", "PubMed"]

def load (dataset: str, motifs: Dict[str, List[int]], target: Optional[Tuple[str, int,  bool]] = None, remove_node_features: bool = False, loader_params: Optional[Dict] = None):
    """Load dataset and compute additional substructure edge index

    Parameters
    ----------
    dataset
        Name of the dataset
    motifs
        Dictinonary containing the names and the sizes of the motifs for which the substructure edge index is computed.
        Available motifs are `ring` and `clique`.
    target, optional
        Replace target by count of a motif with given size (either node_level or not). target = (motif, size, node_level).
    remove_node_features, optional
        Boolean indicating whether node_labels should be removed, by default False
    loader_params, optional
        Dictionary containing train_fraction, val_fraction and batch_size, not needed for Planetoid datasets, by default None.

    Returns
    -------
    The (possibly transformed) dataset, the number of featrues, the number of classes, the number of substructures
    """
    def pre_transform(graph):
        substructures = []
        if "clique" in motifs:
            substructures += [sub.get_cliques(graph.edge_index, max_k = i, min_k = i) for i in motifs["clique"]]
        if "ring" in motifs:
            substructures += [sub.get_rings(graph.edge_index, max_k = i, min_k = i) for i in motifs["ring"]]
        substructures_edge_index = [sub.get_substructure_edge_index(substructure) for substructure in substructures]

        #graph.substructures_edge_index = substructures_edge_index
        update = {"substructures_edge_index": substructures_edge_index}

        if target:
            shape, size, node_level = target
            if shape == "clique":
                y = sub.get_node_counts(sub.get_cliques(graph.edge_index, max_k = size, min_k = size), graph.num_nodes)
            elif shape == "ring":
                y = sub.get_node_counts(sub.get_rings(graph.edge_index, max_k = size, min_k = size), graph.num_nodes)
            elif shape == "triangle+square": # a bit hacky...
                triangle = sub.get_rings(graph.edge_index, max_k = 3, min_k = 3)
                square = sub.get_rings(graph.edge_index, max_k = 4, min_k = 4)
                y = sub.get_node_counts(triangle, graph.num_nodes) + sub.get_node_counts(square, graph.num_nodes)
            else:
                raise RuntimeError(f"Shape {shape} not supported")
            
            if not node_level:
                y = y.sum()
            #graph.y = y
            update["y"] = y
        
        if remove_node_features:
            #graph.x = torch.zeros((graph.num_nodes,1))
            update["x"] = torch.zeros((graph.num_nodes,1))
        
        graph.update(update)
        #return graph.update(update)
        return graph

    if dataset in TU_DATASETS:
        #data = TUDataset(root=f'/tmp/{dataset}_{motifs}_{target}_{remove_node_features}', name= dataset, pre_transform = pre_transform, use_node_attr= True)
        # compute hash to uniquely identify each preprocessing variant
        data = TUDataset(root=f'/tmp/{dataset}_{hash((tuple((key, size) for key, sizes in motifs.items() for size in sizes), tuple(target), remove_node_features))}', name= dataset, pre_transform = pre_transform, use_node_attr= True)
        data = data.shuffle()
        train_num = int(len(data) * loader_params["train_fraction"])
        val_num = int(len(data) * loader_params["val_fraction"])
        train_dataset = data[:train_num]
        val_dataset = data[train_num:train_num + val_num]
        test_dataset = data[train_num + val_num:]
        train_loader = DataLoader(train_dataset, batch_size = loader_params["batch_size"], num_workers = loader_params["num_workers"])
        val_loader = DataLoader(val_dataset, batch_size= loader_params["batch_size"], num_workers = loader_params["num_workers"])
        test_loader = DataLoader(test_dataset, batch_size= loader_params["batch_size"], num_workers = loader_params["num_workers"])
    if dataset in PLANETOID_DATASETS:
        data = Planetoid(root=f'/tmp/{dataset}_{motifs}_{target}_{remove_node_features}', name= dataset, pre_transform = pre_transform)
        # transductive setting
        train_loader = DataLoader(data, batch_size = 1)
        val_loader = DataLoader(data, batch_size= 1)
        test_loader = DataLoader(data, batch_size= 1)
    
    num_features = data.num_features
    num_classes = data.num_classes
    num_substructures = len(data[0].substructures_edge_index)
    return train_loader, val_loader, test_loader, num_features, num_classes, num_substructures
