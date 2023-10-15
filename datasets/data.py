import datasets.fragmentations.fragmentations as frag
from datasets.graph_to_mol import ZINC_Graph_Add_Mol, OGB_Graph_Add_Mol #I have no idea why but this import has to be in the beginning (I know that this is probably a bad sign...)
from typing import Dict, List, Optional, Tuple
from torch_geometric.datasets import TUDataset, Planetoid, ZINC
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torch_geometric.transforms import OneHotDegree, BaseTransform, Compose
from ogb.graphproppred import PygGraphPropPredDataset
from torch.nn.functional import one_hot
import datasets.substructures as sub
import torch
import os.path





TU_DATASETS = ["MUTAG", "ENZYMES", "PROTEINS", "COLLAB", "IMDB-BINARY", "REDDIT-BINARY"]
PLANETOID_DATASETS = ["Cora", "CiteSeer", "PubMed"]

DATASET_ROOT = f"./datasets/data"

def load (dataset: str, motifs: Dict[str, List[int]] = {}, target: List = [], remove_node_features: bool = False, one_hot_degree: bool = False, substructure_node_feature: List = [], loader_params: Optional[Dict] = None):
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
    one_hot_degree, optional
        Boolean indicating whether to concatinate the node features with a one hot encoded node degree, by default False.
    loader_params, optional
        Dictionary containing train_fraction, val_fraction and batch_size, not needed for Planetoid datasets, by default None.

    Returns
    -------
    The (possibly transformed) dataset, the number of featrues, the number of classes, the number of substructures
    """

    one_hot_classes: Optional[int] = None

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
            graph.x = torch.zeros((graph.num_nodes,1))
            #update["x"] = torch.zeros((graph.num_nodes,1))
        
        if one_hot_classes:
            graph.x = one_hot(graph.x, num_classes = one_hot_classes).float().squeeze(1)
        
        if substructure_node_feature:
            shape, size = substructure_node_feature
            if shape == "clique":
                counts = sub.get_node_counts(sub.get_cliques(graph.edge_index, max_k = size, min_k = size), graph.num_nodes)
            elif shape == "ring":
                counts = sub.get_node_counts(sub.get_rings(graph.edge_index, max_k = size, min_k = size), graph.num_nodes)
            else:
                raise RuntimeError(f"Shape {shape} not supported")
            if graph.x.dim == 1:
                graph.x = torch.unsqueeze(graph.x, 1)
            graph.x = torch.cat((graph.x, torch.unsqueeze(counts, 1)), dim =1)
        
        graph.update(update)
        #return graph.update(update)
        return graph

    if dataset in TU_DATASETS:
        #data = TUDataset(root=f'/tmp/{dataset}_{motifs}_{target}_{remove_node_features}', name= dataset, pre_transform = pre_transform, use_node_attr= True)
        # compute hash to uniquely identify each preprocessing variant
        data = TUDataset(root=f'/tmp/{dataset}_{hash((tuple((key, size) for key, sizes in motifs.items() for size in sizes), tuple(target), remove_node_features, tuple(substructure_node_feature)))}', name= dataset, pre_transform = pre_transform, use_node_attr= True)
        data = data.shuffle()

        if one_hot_degree:
            data = OneHotDegree(max_degree = _get_max_degree(data))(data)

        train_num = int(len(data) * loader_params["train_fraction"])
        val_num = int(len(data) * loader_params["val_fraction"])
        train_dataset = data[:train_num]
        val_dataset = data[train_num:train_num + val_num]
        test_dataset = data[train_num + val_num:]
        train_loader = DataLoader(train_dataset, batch_size = loader_params["batch_size"], num_workers = loader_params["num_workers"], shuffle = True)
        val_loader = DataLoader(val_dataset, batch_size= loader_params["batch_size"], num_workers = loader_params["num_workers"])
        test_loader = DataLoader(test_dataset, batch_size= loader_params["batch_size"], num_workers = loader_params["num_workers"])

    elif dataset in PLANETOID_DATASETS:
        data = Planetoid(root=f'/tmp/{dataset}_{motifs}_{target}_{remove_node_features}', name= dataset, pre_transform = pre_transform)

        if one_hot_degree:
            data = OneHotDegree(max_degree = _get_max_degree(data))(data)

        # transductive setting
        train_loader = DataLoader(data, batch_size = 1)
        val_loader = DataLoader(data, batch_size= 1)
        test_loader = DataLoader(data, batch_size= 1)
    
    elif dataset == "ZINC" or dataset == "ZINC-full":
        one_hot_classes = 30
        
        if dataset == "ZINC-full":
            subset = False
        else:
            subset = True
        
        config_hash = hash((tuple((key, size) for key, sizes in motifs.items() for size in sizes), tuple(target), tuple(substructure_node_feature))) + 1

        train_data = ZINC(root=f'/tmp/{dataset}_{config_hash}', pre_transform = pre_transform, subset = subset, split = "train")
        val_data = ZINC(root=f'/tmp/{dataset}_{config_hash}', pre_transform = pre_transform, subset = subset, split = "val")
        test_data = ZINC(root=f'/tmp/{dataset}_{config_hash}', pre_transform = pre_transform, subset = subset, split = "test")
        data = train_data
        
        train_loader = DataLoader(train_data, batch_size = loader_params["batch_size"], num_workers = loader_params["num_workers"])
        val_loader = DataLoader(val_data, batch_size = loader_params["batch_size"], num_workers = loader_params["num_workers"])
        test_loader = DataLoader(test_data, batch_size = loader_params["batch_size"], num_workers = loader_params["num_workers"])
    
    else:
        raise RuntimeError(f"Dataset {dataset} is not supported")
    
    num_features = data.num_features
    num_classes = data.num_classes
    num_substructures = len(data[0].substructures_edge_index)
    return train_loader, val_loader, test_loader, num_features, num_classes, num_substructures

def load_fragmentation(dataset, remove_node_features, one_hot_degree, one_hot_node_features: bool, one_hot_edge_features, fragmentation_method, loader_params):

    if fragmentation_method:
        frag_type, frag_usage, max_vocab = fragmentation_method

        vocab = get_vocab(dataset, frag_type, max_vocab)
        
        frag_constructions = {"BBB": frag.BreakingBridgeBonds, "PSM": frag.PSM, "BRICS": frag.BRICS, "Magnet": frag.Magnet, "Rings": frag.Rings}
        frag_usages = {"node_feature": frag.NodeFeature(max_vocab), "global": frag.GraphLevelFeature(max_vocab), "fragment": frag.FragmentRepresentation(max_vocab)}
    

    #create fragmentation
    transformations = []
    
    if remove_node_features:
        transformations.append(RemoveNodeFeature())
    
    if one_hot_node_features:
        one_hot_classes = 30
        transformations.append(OneHotEncoding(num_classes=one_hot_classes, feature = "x"))
    
    if one_hot_edge_features:
        num_edge_types = 4
        transformations.append(OneHotEncoding(num_classes=num_edge_types, feature = "edge_attr"))

    if one_hot_degree:
        transformations.append(OneHotDegree(100))
    
    if fragmentation_method:
        frag_construction = frag_constructions[frag_type](vocab) if vocab else frag_constructions[frag_type](max_vocab)
        frag_representation = frag_usages[frag_usage]
        transformations.append(frag_construction)
        transformations.append(frag_representation)

    config_hash = hash((remove_node_features, one_hot_edge_features, one_hot_degree, tuple(fragmentation_method)))

    if dataset == "ZINC" or dataset == "ZINC-full":
        transformations.insert(0, ZINC_Graph_Add_Mol())
        transform = Compose(transformations)
        subset = True if dataset == "ZINC" else False
        train_data = ZINC(root=f'{DATASET_ROOT}/{dataset}/{dataset}_{config_hash}', pre_transform = transform, subset = subset, split = "train")
        val_data = ZINC(root=f'{DATASET_ROOT}/{dataset}/{dataset}_{config_hash}', pre_transform = transform, subset = subset, split = "val")
        test_data = ZINC(root=f'{DATASET_ROOT}/{dataset}/{dataset}_{config_hash}', pre_transform = transform, subset = subset, split = "test")
        data = train_data
    
    elif dataset == "OGBG-MOLHIV":
        transformations.insert(0, ZINC_Graph_Add_Mol())
        transform = Compose(transformations)
        data = PygGraphPropPredDataset(name = f"{dataset}_{config_hash}", root = f"{DATASET_ROOT}/{dataset}", pre_transform= transform)
        split_idx = dataset.get_idx_split()
        train_data = data[split_idx["train"]]
        val_data = data[split_idx["valid"]]
        test_data = data[split_idx["test"]]
    
    else:
        raise RuntimeError("Dataset {dataset} not supported")

        
        
    if fragmentation_method and frag_usage == "fragment":
        follow_batch = ["x", "fragments"]
    else:
        follow_batch = None

    train_loader = DataLoader(train_data, batch_size = loader_params["batch_size"], num_workers = loader_params["num_workers"], follow_batch = follow_batch)
    val_loader = DataLoader(val_data, batch_size = loader_params["batch_size"], num_workers = loader_params["num_workers"], follow_batch = follow_batch)
    test_loader = DataLoader(test_data, batch_size = loader_params["batch_size"], num_workers = loader_params["num_workers"], follow_batch = follow_batch)

    num_features = data.num_features
    num_classes = data.num_classes
    return train_loader, val_loader, test_loader, num_features, num_classes

def get_vocab(dataset: str, frag_type: str, max_vocab = 100, root: str = DATASET_ROOT):
    vocab_constructions = {"BBB": frag.BreakingBridgeBondsVocab(max_vocab_size=max_vocab), 
                          "PSM": frag.PrincipalSubgraphVocab(max_vocab_size=max_vocab, vocab_path = f'/tmp/{dataset}_PSM_vocab_{max_vocab}.txt'),
                          "BRICS": frag.BRICSVocab(max_vocab_size = max_vocab),
                          "Magnet": frag.MagnetVocab(max_vocab_size= max_vocab),
                          "Rings": None}

    if frag_type not in vocab_constructions:
        raise RuntimeError("Fragmentation type is not supported")
    
    if vocab_constructions[frag_type]:
        #check if vocab already exists
        vocab_file_name = f"{root}/{dataset}/{dataset}_{frag_type}_vocab_{max_vocab}"
        if os.path.isfile(vocab_file_name):
            vocab = frag.get_vocab_from_file(vocab_file_name)
        else:
            # create vocab
            if dataset == "ZINC" or "ZINC-full":
                subset = True if dataset == "ZINC" else False
                vocab_data = ZINC(root=f'{DATASET_ROOT}/{dataset}/{dataset}_mol', pre_transform = ZINC_Graph_Add_Mol(), subset = subset, split = "train")
            elif dataset == "OGBG-MOLHIV":
                vocab_data = PygGraphPropPredDataset(name = f"{dataset}_mol", root = DATASET_ROOT, pre_transform= OGB_Graph_Add_Mol())
                split_idx = dataset.get_idx_split()
                vocab_data = vocab_data[split_idx["train"]]
            else:
                raise RuntimeError(f"Dataset {dataset} not supported")

            vocab_generator = vocab_constructions[frag_type]
            for data in vocab_data:
                vocab_generator(data)
            vocab = vocab_generator.get_vocab()
            frag.vocab_to_file(vocab_generator, vocab_file_name)
        
        return vocab
    return None

    


def _get_max_degree(data: Data):
    max_degree = 0
    for graph in data:
        degrees = degree(data.edge_index[0], dtype = torch.long)
        max_degree = max(max_degree, degrees.max())
    return max_degree

class RemoveNodeFeature(BaseTransform):
    def __init__(self):
        pass
    
    def __call__(self, graph):
        graph.x = torch.zeros((graph.num_nodes,1))
        return graph

class OneHotEncoding(BaseTransform):
    def __init__(self, num_classes, feature: str):
        self.num_classes = num_classes
        self.feature = feature
    
    def __call__(self, graph):
        one_hot_encoded =one_hot(getattr(graph, self.feature), num_classes = self.num_classes).float().squeeze(1)
        setattr(graph, self.feature, one_hot_encoded)
        return graph

