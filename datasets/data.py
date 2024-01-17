import datasets.fragmentations.fragmentations as frag
from datasets.graph_to_mol import ZINC_Graph_Add_Mol, OGB_Graph_Add_Mol_By_Smiles #I have no idea why but this import has to be in the beginning (I know that this is probably a bad sign...)
from datasets.plogp import FixPlogP
from typing import Dict, List, Optional, Tuple
from torch_geometric.datasets import TUDataset, Planetoid, ZINC
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torch_geometric.transforms import OneHotDegree, BaseTransform, Compose, add_positional_encoding
# from datasets.lrgb import lrgb
from ogb.graphproppred import PygGraphPropPredDataset
from torch.nn.functional import one_hot
from torch.utils.data import random_split
import torch
import os.path

TU_DATASETS = [
    "MUTAG", "ENZYMES", "PROTEINS", "COLLAB", "IMDB-BINARY", "REDDIT-BINARY"
]
PLANETOID_DATASETS = ["Cora", "CiteSeer", "PubMed"]

VOCAB_ROOT = "./datasets/data"
DATASET_ROOT = "/ceph/hdd/students/kempern/datasets"


def load(dataset: str,
         motifs: Dict[str, List[int]] = {},
         target: List = [],
         remove_node_features: bool = False,
         one_hot_degree: bool = False,
         substructure_node_feature: List = [],
         loader_params: Optional[Dict] = None):
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
    import datasets.substructures as sub
    one_hot_classes: Optional[int] = None

    def pre_transform(graph):
        substructures = []
        if "clique" in motifs:
            substructures += [
                sub.get_cliques(graph.edge_index, max_k=i, min_k=i)
                for i in motifs["clique"]
            ]
        if "ring" in motifs:
            substructures += [
                sub.get_rings(graph.edge_index, max_k=i, min_k=i)
                for i in motifs["ring"]
            ]
        substructures_edge_index = [
            sub.get_substructure_edge_index(substructure)
            for substructure in substructures
        ]

        #graph.substructures_edge_index = substructures_edge_index
        update = {"substructures_edge_index": substructures_edge_index}

        if target:
            shape, size, node_level = target
            if shape == "clique":
                y = sub.get_node_counts(
                    sub.get_cliques(graph.edge_index, max_k=size, min_k=size),
                    graph.num_nodes)
            elif shape == "ring":
                y = sub.get_node_counts(
                    sub.get_rings(graph.edge_index, max_k=size, min_k=size),
                    graph.num_nodes)
            elif shape == "triangle+square":  # a bit hacky...
                triangle = sub.get_rings(graph.edge_index, max_k=3, min_k=3)
                square = sub.get_rings(graph.edge_index, max_k=4, min_k=4)
                y = sub.get_node_counts(triangle,
                                        graph.num_nodes) + sub.get_node_counts(
                                            square, graph.num_nodes)
            else:
                raise RuntimeError(f"Shape {shape} not supported")

            if not node_level:
                y = y.sum()
            #graph.y = y
            update["y"] = y

        if remove_node_features:
            graph.x = torch.zeros((graph.num_nodes, 1))
            #update["x"] = torch.zeros((graph.num_nodes,1))

        if one_hot_classes:
            graph.x = one_hot(graph.x,
                              num_classes=one_hot_classes).float().squeeze(1)

        if substructure_node_feature:
            shape, size = substructure_node_feature
            if shape == "clique":
                counts = sub.get_node_counts(
                    sub.get_cliques(graph.edge_index, max_k=size, min_k=size),
                    graph.num_nodes)
            elif shape == "ring":
                counts = sub.get_node_counts(
                    sub.get_rings(graph.edge_index, max_k=size, min_k=size),
                    graph.num_nodes)
            else:
                raise RuntimeError(f"Shape {shape} not supported")
            if graph.x.dim == 1:
                graph.x = torch.unsqueeze(graph.x, 1)
            graph.x = torch.cat((graph.x, torch.unsqueeze(counts, 1)), dim=1)

        graph.update(update)
        #return graph.update(update)
        return graph

    if dataset in TU_DATASETS:
        #data = TUDataset(root=f'/tmp/{dataset}_{motifs}_{target}_{remove_node_features}', name= dataset, pre_transform = pre_transform, use_node_attr= True)
        # compute hash to uniquely identify each preprocessing variant
        data = TUDataset(
            root=
            f'/tmp/{dataset}_{hash((tuple((key, size) for key, sizes in motifs.items() for size in sizes), tuple(target), remove_node_features, tuple(substructure_node_feature)))}',
            name=dataset,
            pre_transform=pre_transform,
            use_node_attr=True)
        data = data.shuffle()

        if one_hot_degree:
            data = OneHotDegree(max_degree=_get_max_degree(data))(data)

        train_num = int(len(data) * loader_params["train_fraction"])
        val_num = int(len(data) * loader_params["val_fraction"])
        train_dataset = data[:train_num]
        val_dataset = data[train_num:train_num + val_num]
        test_dataset = data[train_num + val_num:]
        train_loader = DataLoader(train_dataset,
                                  batch_size=loader_params["batch_size"],
                                  num_workers=loader_params["num_workers"],
                                  shuffle=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=loader_params["batch_size"],
                                num_workers=loader_params["num_workers"])
        test_loader = DataLoader(test_dataset,
                                 batch_size=loader_params["batch_size"],
                                 num_workers=loader_params["num_workers"])

    elif dataset in PLANETOID_DATASETS:
        data = Planetoid(
            root=f'/tmp/{dataset}_{motifs}_{target}_{remove_node_features}',
            name=dataset,
            pre_transform=pre_transform)

        if one_hot_degree:
            data = OneHotDegree(max_degree=_get_max_degree(data))(data)

        # transductive setting
        train_loader = DataLoader(data, batch_size=1)
        val_loader = DataLoader(data, batch_size=1)
        test_loader = DataLoader(data, batch_size=1)

    elif dataset == "ZINC" or dataset == "ZINC-full":
        one_hot_classes = 30

        if dataset == "ZINC-full":
            subset = False
        else:
            subset = True

        config_hash = hash((tuple((key, size) for key, sizes in motifs.items()
                                  for size in sizes), tuple(target),
                            tuple(substructure_node_feature))) + 1

        train_data = ZINC(root=f'/tmp/{dataset}_{config_hash}',
                          pre_transform=pre_transform,
                          subset=subset,
                          split="train")
        val_data = ZINC(root=f'/tmp/{dataset}_{config_hash}',
                        pre_transform=pre_transform,
                        subset=subset,
                        split="val")
        test_data = ZINC(root=f'/tmp/{dataset}_{config_hash}',
                         pre_transform=pre_transform,
                         subset=subset,
                         split="test")
        data = train_data
        data.num_classes = 1

        train_loader = DataLoader(train_data,
                                  batch_size=loader_params["batch_size"],
                                  num_workers=loader_params["num_workers"])
        val_loader = DataLoader(val_data,
                                batch_size=loader_params["batch_size"],
                                num_workers=loader_params["num_workers"])
        test_loader = DataLoader(test_data,
                                 batch_size=loader_params["batch_size"],
                                 num_workers=loader_params["num_workers"])

    else:
        raise RuntimeError(f"Dataset {dataset} is not supported")

    num_features = data.num_features
    num_classes = data.num_classes
    num_substructures = len(data[0].substructures_edge_index)
    return train_loader, val_loader, test_loader, num_features, num_classes, num_substructures


def load_fragmentation(dataset,
                       remove_node_features,
                       one_hot_degree,
                       one_hot_node_features: bool,
                       one_hot_edge_features,
                       fragmentation_method,
                       loader_params,
                       encoding=None,
                       subset_frac=1,
                       higher_edge_features=False,
                       dataset_seed=None):

    if fragmentation_method:
        frag_type, frag_usage, vocab_size_params = fragmentation_method

        if frag_usage == "higher_level_graph_tree":
            # later on, tree construction introduces an additional junction fragment
            vocab_size_params = vocab_size_params.copy(
            )  # allows us to change contents
            vocab_size_params["vocab_size"] -= 1

        vocab = get_vocab(dataset, frag_type, vocab_size_params["vocab_size"])

        frag_constructions = {
            "BBB": frag.BreakingBridgeBonds,
            "PSM": frag.PSM,
            "BRICS": frag.BRICS,
            "Magnet": frag.Magnet,
            "Rings": frag.Rings,
            "RingsEdges": frag.RingsEdges,
            "RingsPaths": frag.RingsPaths,
            "MagnetWithoutVocab": frag.MagnetWithoutVocab
        }
        frag_usages = {
            "node_feature":
            frag.NodeFeature(vocab_size_params["vocab_size"]),
            "global":
            frag.GraphLevelFeature(vocab_size_params["vocab_size"]),
            "fragment":
            frag.FragmentRepresentation(vocab_size_params["vocab_size"]),
            "higher_level_graph_tree":
            frag.HigherLevelGraph(vocab_size_params["vocab_size"],
                                  "tree",
                                  higher_edge_features=higher_edge_features),
            "higher_level_graph_node":
            frag.HigherLevelGraph(vocab_size_params["vocab_size"],
                                  "node",
                                  higher_edge_features=higher_edge_features)
        }

    #create fragmentation
    transformations = []

    if remove_node_features:
        transformations.append(RemoveNodeFeature())

    if one_hot_node_features:
        one_hot_classes = 30
        transformations.append(
            OneHotEncoding(num_classes=one_hot_classes, feature="x"))

    if one_hot_edge_features:
        num_edge_types = 4
        transformations.append(
            OneHotEncoding(num_classes=num_edge_types, feature="edge_attr"))

    if one_hot_degree:
        transformations.append(OneHotDegree(100))

    if fragmentation_method:
        frag_construction = frag_constructions[frag_type](
            vocab) if vocab else frag_constructions[frag_type](
                **vocab_size_params)
        frag_representation = frag_usages[frag_usage]
        transformations.append(frag_construction)
        transformations.append(frag_representation)

    if encoding:
        for encod in encoding:
            if encod["name"] == "random-walk":
                random_walk_encoding = add_positional_encoding.AddRandomWalkPE(
                    encod["walk_length"])
                transformations.append(random_walk_encoding)
            else:
                raise RuntimeError(f"Encoding {encod['name']} not supported.")

    config_name = f"{str(fragmentation_method)}_{str(encoding)}_{str(remove_node_features)}_{str(one_hot_edge_features)}_{str(one_hot_degree)}_{str(one_hot_node_features)}_{str(higher_edge_features)}"

    if dataset == "ZINC" or dataset == "ZINC-full":
        transformations.insert(0, ZINC_Graph_Add_Mol())
        transform = Compose(transformations)
        subset = True if dataset == "ZINC" else False
        train_data = ZINC(
            root=f'{DATASET_ROOT}/{dataset}/{dataset}_{config_name}',
            pre_transform=transform,
            subset=subset,
            split="train")
        val_data = ZINC(
            root=f'{DATASET_ROOT}/{dataset}/{dataset}_{config_name}',
            pre_transform=transform,
            subset=subset,
            split="val")
        test_data = ZINC(
            root=f'{DATASET_ROOT}/{dataset}/{dataset}_{config_name}',
            pre_transform=transform,
            subset=subset,
            split="test")
        data = train_data
        num_classes = 1
    
    elif dataset == "ZINC-fixed" or dataset == "ZINC-full-fixed":   
        transformations.insert(0, ZINC_Graph_Add_Mol())
        transformations.append(FixPlogP())
        transform = Compose(transformations)
        subset = True if dataset == "ZINC" else False
        train_data = ZINC(root=f'{DATASET_ROOT}/{dataset}/{dataset}_{config_name}', pre_transform = transform, subset = subset, split = "train")
        val_data = ZINC(root=f'{DATASET_ROOT}/{dataset}/{dataset}_{config_name}', pre_transform = transform, subset = subset, split = "val")
        test_data = ZINC(root=f'{DATASET_ROOT}/{dataset}/{dataset}_{config_name}', pre_transform = transform, subset = subset, split = "test")
        data = train_data
        num_classes = 1
    
    elif dataset == "ogbg-molhiv":
        transformations.insert(0, OGB_Graph_Add_Mol_By_Smiles())
        transform = Compose(transformations)
        data = PygGraphPropPredDataset(
            name=dataset,
            root=f"{DATASET_ROOT}/{dataset}/{dataset}_{config_name}",
            pre_transform=transform)
        split_idx = data.get_idx_split()
        train_data = data[split_idx["train"]]
        val_data = data[split_idx["valid"]]
        test_data = data[split_idx["test"]]
        num_classes = data.num_classes

    elif dataset == "peptides-struct":
        transform = Compose(transformations)
        data = lrgb.PeptidesStructuralDataset(
            root=f"{DATASET_ROOT}/{dataset}/{dataset}_{config_name}",
            pre_transform=transform,
            smiles2graph=lrgb.smiles2graph_add_mol)
        split_idx = data.get_idx_split()
        train_data = data[split_idx["train"]]
        val_data = data[split_idx["val"]]
        test_data = data[split_idx["test"]]
        num_classes = data.num_classes

    elif dataset == "peptides-func":
        transform = Compose(transformations)
        data = lrgb.PeptidesFunctionalDataset(
            root=f"{DATASET_ROOT}/{dataset}/{dataset}_{config_name}",
            pre_transform=transform,
            smiles2graph=lrgb.smiles2graph_add_mol)
        split_idx = data.get_idx_split()
        train_data = data[split_idx["train"]]
        val_data = data[split_idx["val"]]
        test_data = data[split_idx["test"]]
        num_classes = 2  # multilabel binary classification task

    else:
        raise RuntimeError(f"Dataset {dataset} not supported")

    if fragmentation_method and frag_usage in [
            "fragment", "higher_level_graph_tree", "higher_level_graph_node"
    ]:
        follow_batch = ["x", "fragments"]
    else:
        follow_batch = None

    if subset_frac < 1:
        subset_size = int(subset_frac * len(train_data))
        train_data, _ = random_split(
            train_data,
            [subset_size, len(train_data) - subset_size])

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

    num_features = data.num_features
    return train_loader, val_loader, test_loader, num_features, num_classes


def get_vocab(dataset: str,
              frag_type: str,
              max_vocab=100,
              root: str = VOCAB_ROOT):
    vocab_constructions = {
        "BBB":
        frag.BreakingBridgeBondsVocab(vocab_size=max_vocab),
        "PSM":
        frag.PrincipalSubgraphVocab(
            vocab_size=max_vocab,
            vocab_path=f'/tmp/{dataset}_PSM_vocab_{max_vocab}.txt'),
        "BRICS":
        frag.BRICSVocab(vocab_size=max_vocab),
        "Magnet":
        frag.MagnetVocab(vocab_size=max_vocab),
        "Rings":
        None,
        "RingsEdges":
        None,
        "RingsPaths":
        None,
        "MagnetWithoutVocab":
        None
    }

    if frag_type not in vocab_constructions:
        raise RuntimeError("Fragmentation type is not supported")

    if vocab_constructions[frag_type]:
        #check if vocab already exists
        vocab_file_name = f"{root}/{dataset}/{dataset}_{frag_type}_vocab_{max_vocab}"
        if os.path.isfile(vocab_file_name):
            vocab = frag.get_vocab_from_file(vocab_file_name)
        else:
            # create vocab
            if dataset == "ZINC" or dataset == "ZINC-full":
                subset = True if dataset == "ZINC" else False
                vocab_data = ZINC(root=f'{root}/{dataset}/{dataset}_mol',
                                  pre_transform=ZINC_Graph_Add_Mol(),
                                  subset=subset,
                                  split="train")
            elif dataset == "ogbg-molhiv":
                vocab_data = PygGraphPropPredDataset(
                    name=dataset,
                    root=f'{root}/{dataset}/{dataset}_mol',
                    pre_transform=OGB_Graph_Add_Mol_By_Smiles())
                split_idx = vocab_data.get_idx_split()
                vocab_data = vocab_data[split_idx["train"]]
            elif dataset == "peptides-struct":
                vocab_data = lrgb.PeptidesStructuralDataset(
                    root=f'{root}/{dataset}/{dataset}_mol',
                    smiles2graph=lrgb.smiles2graph_add_mol)
                split_idx = vocab_data.get_idx_split()
                vocab_data = vocab_data[split_idx["train"]]
            elif dataset == "peptides-func":
                vocab_data = lrgb.PeptidesFunctionalDataset(
                    root=f'{root}/{dataset}/{dataset}_mol',
                    smiles2graph=lrgb.smiles2graph_add_mol)
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
            getattr(graph, self.feature),
            num_classes=self.num_classes).float().squeeze(1)
        setattr(graph, self.feature, one_hot_encoded)
        return graph
