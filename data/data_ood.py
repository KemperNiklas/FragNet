import os.path
from typing import Dict, List, Optional, Tuple

import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch.nn.functional import one_hot
from torch_geometric.data import Data
from torch_geometric.datasets import ZINC, Planetoid, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import BaseTransform, Compose, OneHotDegree
from torch_geometric.utils import degree

import data.fragmentations.fragmentations as frag
from config import DATASET_ROOT, VOCAB_ROOT
from data.graph_to_mol import (  # I have no idea why but this import has to be in the beginning (I know that this is probably a bad sign...)
    OGB_Graph_Add_Mol_By_Smiles, ZINC_Graph_Add_Mol)


def load_fragmentation(dataset, remove_node_features, one_hot_degree, one_hot_node_features: bool, one_hot_edge_features, fragmentation_method, loader_params, filter):
    """Like the load_fragmentation function in data/data.py but additionally allows to filter out fragments of size filter
    """

    if fragmentation_method:
        frag_type, frag_usage, vocab_size_params = fragmentation_method

        if frag_usage == "higher_level_graph_tree":
            # later on, tree construction introduces an additional junction fragment
            vocab_size_params = vocab_size_params.copy()  # allows us to change contents
            vocab_size_params["vocab_size"] -= 1

        vocab = get_vocab(dataset, frag_type, vocab_size_params["vocab_size"])

        frag_constructions = {"BBB": frag.BreakingBridgeBonds, "PSM": frag.PSM, "BRICS": frag.BRICS,
                              "Magnet": frag.Magnet, "Rings": frag.Rings, "RingsEdges": frag.RingsEdges, "RingsPaths": frag.RingsPaths}
        frag_usages = {"node_feature": frag.NodeFeature(vocab_size_params["vocab_size"]), "global": frag.GraphLevelFeature(vocab_size_params["vocab_size"]), "fragment": frag.FragmentRepresentation(
            vocab_size_params["vocab_size"]), "higher_level_graph_tree": frag.HigherLevelGraph(vocab_size_params["vocab_size"], "tree"), "higher_level_graph_node": frag.HigherLevelGraph(vocab_size_params["vocab_size"], "node")}

    # create fragmentation
    transformations = []

    if remove_node_features:
        transformations.append(RemoveNodeFeature())

    if one_hot_node_features:
        one_hot_classes = 30
        transformations.append(OneHotEncoding(
            num_classes=one_hot_classes, feature="x"))

    if one_hot_edge_features:
        num_edge_types = 4
        transformations.append(OneHotEncoding(
            num_classes=num_edge_types, feature="edge_attr"))

    if one_hot_degree:
        transformations.append(OneHotDegree(100))

    if fragmentation_method:
        frag_construction = frag_constructions[frag_type](
            vocab) if vocab else frag_constructions[frag_type](**vocab_size_params)
        frag_representation = frag_usages[frag_usage]
        transformations.append(frag_construction)
        transformations.append(frag_representation)

    if fragmentation_method:
        config_hash = hash((remove_node_features, one_hot_edge_features, one_hot_degree,
                           fragmentation_method[0], fragmentation_method[1], tuple(fragmentation_method[2].items())))
    else:
        config_hash = hash(
            (remove_node_features, one_hot_edge_features, one_hot_degree))

    if dataset == "ZINC" or dataset == "ZINC-full":
        transformations.insert(0, ZINC_Graph_Add_Mol())
        transform = Compose(transformations)
        subset = True if dataset == "ZINC" else False
        train_data = ZINC(root=f'{DATASET_ROOT}/{dataset}/{dataset}_{config_hash}',
                          pre_transform=transform, subset=subset, split="train")
        data = train_data
        train_data = [a for a in train_data if all(
            a.fragment_types[:, 1] != filter)]
        val_data = ZINC(root=f'{DATASET_ROOT}/{dataset}/{dataset}_{config_hash}',
                        pre_transform=transform, subset=subset, split="val")
        test_data = ZINC(root=f'{DATASET_ROOT}/{dataset}/{dataset}_{config_hash}',
                         pre_transform=transform, subset=subset, split="test")
        num_classes = 1

    elif dataset == "ogbg-molhiv":
        transformations.insert(0, OGB_Graph_Add_Mol_By_Smiles())
        transform = Compose(transformations)
        data = PygGraphPropPredDataset(
            name=dataset, root=f"{DATASET_ROOT}/{dataset}/{dataset}_{config_hash}", pre_transform=transform)
        split_idx = data.get_idx_split()
        train_data = data[split_idx["train"]]
        val_data = data[split_idx["valid"]]
        test_data = data[split_idx["test"]]
        num_classes = data.num_classes

    else:
        raise RuntimeError(f"Dataset {dataset} not supported")

    if fragmentation_method and frag_usage in ["fragment", "higher_level_graph_tree", "higher_level_graph_node"]:
        follow_batch = ["x", "fragments"]
    else:
        follow_batch = None

    train_loader = DataLoader(train_data, batch_size=loader_params["batch_size"],
                              num_workers=loader_params["num_workers"], follow_batch=follow_batch, shuffle=True)
    val_batch_size = loader_params["val_batch_size"] if "val_batch_size" in loader_params else loader_params["batch_size"]
    val_loader = DataLoader(val_data, batch_size=val_batch_size,
                            num_workers=loader_params["num_workers"], follow_batch=follow_batch)
    test_loader = DataLoader(test_data, batch_size=val_batch_size,
                             num_workers=loader_params["num_workers"], follow_batch=follow_batch)

    num_features = data.num_features
    return train_loader, val_loader, test_loader, num_features, num_classes


def get_vocab(dataset: str, frag_type: str, max_vocab=100, root: str = VOCAB_ROOT):
    vocab_constructions = {"BBB": frag.BreakingBridgeBondsVocab(vocab_size=max_vocab),
                           "PSM": frag.PrincipalSubgraphVocab(vocab_size=max_vocab, vocab_path=f'/tmp/{dataset}_PSM_vocab_{max_vocab}.txt'),
                           "BRICS": frag.BRICSVocab(vocab_size=max_vocab),
                           "Magnet": frag.MagnetVocab(vocab_size=max_vocab),
                           "Rings": None,
                           "RingsEdges": None,
                           "RingsPaths": None}

    if frag_type not in vocab_constructions:
        raise RuntimeError("Fragmentation type is not supported")

    if vocab_constructions[frag_type]:
        # check if vocab already exists
        vocab_file_name = f"{root}/{dataset}/{dataset}_{frag_type}_vocab_{max_vocab}"
        if os.path.isfile(vocab_file_name):
            vocab = frag.get_vocab_from_file(vocab_file_name)
        else:
            # create vocab
            if dataset == "ZINC" or dataset == "ZINC-full":
                subset = True if dataset == "ZINC" else False
                vocab_data = ZINC(root=f'{root}/{dataset}/{dataset}_mol',
                                  pre_transform=ZINC_Graph_Add_Mol(), subset=subset, split="train")
            elif dataset == "ogbg-molhiv":
                vocab_data = PygGraphPropPredDataset(
                    name=dataset, root=f'{root}/{dataset}/{dataset}_mol', pre_transform=OGB_Graph_Add_Mol_By_Smiles())
                split_idx = vocab_data.get_idx_split()
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
        degrees = degree(data.edge_index[0], dtype=torch.long)
        max_degree = max(max_degree, degrees.max())
    return max_degree


class RemoveNodeFeature(BaseTransform):
    def __init__(self):
        pass

    def __call__(self, graph):
        graph.x = torch.zeros((graph.num_nodes, 1))
        return graph


class OneHotEncoding(BaseTransform):
    def __init__(self, num_classes, feature: str):
        self.num_classes = num_classes
        self.feature = feature

    def __call__(self, graph):
        one_hot_encoded = one_hot(
            getattr(graph, self.feature), num_classes=self.num_classes).float().squeeze(1)
        setattr(graph, self.feature, one_hot_encoded)
        return graph
