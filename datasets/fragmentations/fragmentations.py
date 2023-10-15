from datasets.fragmentations.magnet import MolDecomposition
from datasets.fragmentations.breaking_bridge_bonds import find_motifs_from_vocabulary, MotifExtractionSettings, MotifVocabularyExtractor
from datasets.fragmentations.mol_bpe import Tokenizer, graph_bpe_smiles
from collections import Counter
from rdkit.Chem.BRICS import BRICSDecompose, BreakBRICSBonds
from rdkit import Chem
from typing import Any, List
from rdkit.Chem.rdmolops import GetMolFrags
from torch_geometric.transforms import BaseTransform
import torch
from torch_geometric.data import Data
import pickle

def vocab_to_file(vocab, file_name):
    if not file_name:
        file_name = f"./{vocab.__name__}_vocab_{vocab.max_vocab_size}"
    pickle.dump(vocab.get_vocab(), open(file_name, "wb"))

def get_vocab_from_file(file_name):
    return pickle.load(open(file_name, "rb"))

class BreakingBridgeBondsVocab(BaseTransform):
    def __init__(self, min_frequency = None, min_num_atoms = 3, cut_leaf_edges = False, max_vocab_size = 200):
        settings = MotifExtractionSettings(min_frequency = min_frequency, min_num_atoms=min_num_atoms, cut_leaf_edges=cut_leaf_edges, max_vocab_size=max_vocab_size)
        self.extractor = MotifVocabularyExtractor(settings)
        self.max_vocab_size = max_vocab_size
    
    def __call__(self, graph):
        mol = graph.mol
        self.extractor.update(mol)
        return graph
    
    def get_vocab(self):
        return self.extractor.output()
    

class BRICSVocab(BaseTransform):
    def __init__(self, max_vocab_size = 200):
        self.max_vocab_size = max_vocab_size
        self.counter = Counter()
    
    def __call__(self, graph):
        mol = graph.mol
        fragments = [Chem.MolToSmiles(fragment) for fragment in GetMolFrags(BreakBRICSBonds(mol), asMols = True)]
        #filter fragments with only one atom
        #TODO
        self.counter.update(fragments)
        return graph
    
    def get_vocab(self):
        return [motif for motif,_ in self.counter.most_common(self.max_vocab_size)]


class PrincipalSubgraphVocab(BaseTransform):
    def __init__(self, max_vocab_size = 200, vocab_path = "./principal_subgraph_vocab.txt", cpus = 4, kekulize = False):
        self.max_vocab_size = max_vocab_size
        self.smis = []
        self.vocab_path = vocab_path
        self.cpus = cpus
        self.kekulize = kekulize
    
    def __call__(self, graph):
        self.smis.append(Chem.MolToSmiles(graph.mol))
        return graph

    def get_vocab(self):
        graph_bpe_smiles(self.smis, vocab_len = self.max_vocab_size, vocab_path = self.vocab_path, cpus = self.cpus, kekulize = self.kekulize)
        return self.vocab_path

class MagnetVocab(BaseTransform):
    def __init__(self, max_vocab_size = 200):
        self.max_vocab_size = max_vocab_size
        self.hash_counter = Counter()
    
    def __call__(self, graph):
        mol = Chem.Mol(graph.mol) #create copy of molecule
        hashes = MolDecomposition(mol).id_to_hash.values()
        self.hash_counter.update(hashes)
    
    def get_vocab(self):
        return [hash for (hash, _) in self.hash_counter.most_common(self.max_vocab_size)]
    

class Magnet(BaseTransform):
    def __init__(self, vocab):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.hash_to_index = {hash: id for id, hash in enumerate(self.vocab)}
    
    def __call__(self, graph):
        mol = Chem.Mol(graph.mol) #create copy of molecule
        decomposition = MolDecomposition(mol)
        ids_in_vocab = [id for (id, hash) in decomposition.id_to_hash.items() if hash in self.vocab]
        
        node_substructures = []
        fragment_to_index = {}
        for fragments in decomposition.nodes.values():
            fragment_info = []
            for frag_id in fragments:
                if frag_id in ids_in_vocab:
                    hash = decomposition.id_to_hash[frag_id]
                    if frag_id not in fragment_to_index:
                        fragment_to_index[frag_id] = len(fragment_to_index)
                    fragment_info.append((fragment_to_index[frag_id], self.hash_to_index[hash]))
            node_substructures.append(fragment_info)
        
        graph.substructures = node_substructures
        return graph

class BRICS(BaseTransform):
    def __init__(self, vocab: List[str]):
        self.vocab = vocab
        self.vocab_size = len(vocab)
    
    def __call__(self, graph):
        mol = graph.mol
        node_substructures = [[] for _ in range(graph.num_nodes)]

        fragments = GetMolFrags(BreakBRICSBonds(mol), asMols = True)
        fragments_atom_ids = GetMolFrags(BreakBRICSBonds(mol))

        fragment_id = 0
        for fragment, atom_ids in zip(fragments, fragments_atom_ids):
            if Chem.MolToSmiles(fragment) in self.vocab:
                fragment_type = self.vocab.index(Chem.MolToSmiles(fragment))
                #filter atom ids that are not introduced by BRICS
                atom_ids_filtered = [atom_id for atom_id in atom_ids if atom_id < graph.num_nodes]

                for id in atom_ids_filtered:
                    node_substructures[id].append((fragment_id, fragment_type))

                fragment_id += 1
                
        graph.substructures = node_substructures
        return graph

class BreakingBridgeBonds(BaseTransform):
    def __init__(self, vocab) -> None:
        self.vocab = vocab
        self.vocab_size = len(vocab.vocabulary)

    def __call__(self, graph):
        mol = graph.mol
        node_substructures = [[] for _ in range(graph.num_nodes)]
        for fragment_id, fragment in enumerate(find_motifs_from_vocabulary(mol, self.vocab)):
            fragment_type = self.vocab.vocabulary[fragment.motif_type] 
            atoms = [atom.atom_id for atom in fragment.atoms]
            for atom in atoms:
                node_substructures[atom].append((fragment_id, fragment_type))
        graph.substructures = node_substructures
        return graph

class PSM(BaseTransform):
    def __init__(self, vocab: str) -> None:
        self.vocab = vocab
        self.tokenizer = Tokenizer(self.vocab)
        self.vocab_size = len(self.tokenizer.idx2subgraph) - 2 #contains two special symbols
    
    def __call__(self, graph):
        subgraph_mol = self.tokenizer(graph.mol)
        node_substructures = [[] for _ in range(graph.num_nodes)]
        for fragment_id,fragment in enumerate(subgraph_mol.nodes):
            atom_mapping =  subgraph_mol.get_node(fragment).get_atom_mapping()
            atom_ids = list(atom_mapping.keys())
            fragment_type = self.tokenizer.subgraph2idx[subgraph_mol.get_node(fragment).smiles]
            for atom in atom_ids:
                node_substructures[atom].append((fragment_id, fragment_type))
        graph.substructures = node_substructures
        return graph


    
class Rings(BaseTransform):
    def __init__(self, max_vocab_size = 15) -> None:
        """Initialize ring based fragmentation that finds a set of short Rings 
        covering all rings of the molecule.

        Parameters
        ----------
        max_vocab_size, optional
            Maximum vocab size, i.e. size of the longest ring - 2, by default 15
        """
        self.max_vocab_size = max_vocab_size
        self.max_ring_size = max_vocab_size + 2
    
    def __call__(self, graph):
        mol = graph.mol
        rings = Chem.GetSymmSSSR(mol)
        node_substructures = [[] for _ in range(graph.num_nodes)]
        fragment_id = 0
        for i in range(len(rings)):
            ring = list(rings[i])
            if len(ring) <= self.max_ring_size:
                for atom in ring:
                    fragment_type = len(ring) - 3
                    node_substructures[atom].append((fragment_id, fragment_type))
                fragment_id += 1
        graph.substructures = node_substructures
        return graph


class NodeFeature(BaseTransform):
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
    
    def __call__(self, graph):
        node_features = torch.zeros(graph.num_nodes, self.vocab_size)
        for atom_id, fragments in enumerate(graph.substructures):
            for (_ ,fragment_type) in fragments:
                node_features[atom_id, fragment_type] += 1
        graph.x = torch.cat([graph.x, node_features], dim = 1)
        return graph

class GraphLevelFeature(BaseTransform):
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
    
    def __call__(self, graph):
        graph_features = torch.zeros(self.vocab_size)
        frag_id_to_type = dict([frag_info for frag_infos in graph.substructures for frag_info in frag_infos if frag_info])
        max_frag_id = max([frag_id for frag_infos in graph.substructures for (frag_id, _) in frag_infos], default = -1)
        for id in range(max_frag_id +1):
            graph_features[frag_id_to_type[id]] += 1
        node_counts = torch.sum(graph.x, dim = 0)
        graph.motif_counts = torch.unsqueeze(torch.concat([graph_features, node_counts], dim = 0), dim = 0)   
        return graph

class FragmentData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'fragments_edge_index':
            return torch.tensor([[self.x.size(0)], [self.fragments.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)
    
class FragmentRepresentation(BaseTransform):
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
    
    def __call__(self, graph):
        frag_id_to_type = dict([frag_info for frag_infos in graph.substructures for frag_info in frag_infos if frag_info])
        max_frag_id = max([frag_id for frag_infos in graph.substructures for (frag_id, _) in frag_infos], default = -1)
        frag_representation = torch.zeros(max_frag_id +1, self.vocab_size)
        frag_representation[list(range(max_frag_id +1)), [frag_id_to_type[frag_id] for frag_id in range(max_frag_id +1)]] = 1
        graph.fragments = frag_representation
        edges = [[node_id, frag_id] for node_id, frag_infos in enumerate(graph.substructures) for (frag_id, _) in frag_infos]
        graph.fragments_edge_index = torch.tensor(edges, dtype = torch.long).T.contiguous()
        return FragmentData(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, y= graph.y, mol= graph.mol, substructures= graph.substructures, fragments= graph.fragments, fragments_edge_index= graph.fragments_edge_index)

"""the following is outdated!!!"""
class BRICSGraphLevel(BaseTransform):
    def __init__(self, vocab: List[str]):
        self.vocab = vocab
        self.vocab_size = len(vocab)
    
    def __call__(self, graph):
        mol = Chem.MolFromSmiles(graph.smiles)
        motif_counts = torch.zeros(self.vocab_size)
        fragments = [Chem.MolToSmiles(fragment) for fragment in GetMolFrags(BreakBRICSBonds(mol), asMols = True)]
        motif_ids = [self.vocab.index(fragment) for fragment in fragments if fragment in self.vocab]
        for motif_id in motif_ids:
            motif_counts[motif_id] += 1
        node_counts = torch.sum(graph.x, dim = 0)
        graph.motif_counts = torch.unsqueeze(torch.concat([motif_counts, node_counts], dim = 0), dim = 0)
        return graph

class BRICSNodeFeature(BaseTransform):
    def __init__(self, vocab: List[str]):
        self.vocab = vocab
        self.vocab_size = len(vocab)
    
    def __call__(self, graph):
        mol = Chem.MolFromSmiles(graph.smiles)
        node_features = torch.zeros(graph.num_nodes, self.vocab_size)

        fragments = GetMolFrags(BreakBRICSBonds(mol), asMols = True)
        fragments_atom_ids = GetMolFrags(BreakBRICSBonds(mol))
        for fragment, atom_ids in zip(fragments, fragments_atom_ids):
            if Chem.MolToSmiles(fragment) in self.vocab:
                fragment_id = self.vocab.index(Chem.MolToSmiles(fragment))
                #filter atom ids that are not introduced by BRICS
                atom_ids_filtered = [atom_id for atom_id in atom_ids if atom_id < graph.num_nodes]
                node_features[atom_ids_filtered, fragment_id] = 1
                
        graph.x = torch.cat([graph.x, node_features], dim = 1)
        return graph


class BreakingBridgeBondsGraphLevel(BaseTransform):
    def __init__(self, vocab) -> None:
        self.vocab = vocab
        self.vocab_size = len(vocab.vocabulary)
    
    def __call__(self, graph):
        mol = Chem.MolFromSmiles(graph.smiles)
        motifs = [self.vocab.vocabulary[fragment.motif_type] 
                  for fragment in find_motifs_from_vocabulary(mol, self.vocab)]
        motif_counts = torch.zeros(self.vocab_size)
        for motif_id in motifs:
            motif_counts[motif_id] += 1
        node_counts = torch.sum(graph.x, dim = 0)
        graph.motif_counts = torch.unsqueeze(torch.concat([motif_counts, node_counts], dim = 0), dim = 0)
        return graph
    
class BreakingBridgeBondsNodeFeature(BaseTransform):
    def __init__(self, vocab) -> None:
        self.vocab = vocab
        self.vocab_size = len(vocab.vocabulary)
    
    def __call__(self, graph):
        mol = Chem.MolFromSmiles(graph.smiles)
        node_features = torch.zeros(graph.num_nodes, self.vocab_size)
        for fragment in find_motifs_from_vocabulary(mol, self.vocab):
            feature_id = self.vocab.vocabulary[fragment.motif_type] 
            atoms = [atom.atom_id for atom in fragment.atoms]
            node_features[atoms, feature_id] = 1
        graph.x = torch.cat([graph.x, node_features], dim = 1)
        return graph
    
class RingNodeFeature(BaseTransform):
    def __init__(self) -> None:
        pass

    def __call__(self, graph):
        mol = Chem.MolFromSmiles(graph.smiles)
        node_features = torch.zeros(graph.num_nodes, 1)
        in_ring = [atom.IsInRing() for atom in mol.GetAtoms()]
        node_features[in_ring, 0] = 1
        graph.x = torch.cat([graph.x, node_features], dim = 1)
        return graph

class RingSizeNodeFeature(BaseTransform):
    def __init__(self, max_size = 15) -> None:
        self.max_size = max_size

    def __call__(self, graph):
        mol = Chem.MolFromSmiles(graph.smiles)
        node_features = torch.zeros(graph.num_nodes, self.max_size - 2)
        rings = mol.GetRingInfo().AtomRings()
        for ring in rings:
            if len(ring) <= self.max_size:
                node_features[ring, len(ring) - 3] += 1
            
        graph.x = torch.cat([graph.x, node_features], dim = 1)
        return graph

class RingGraphLevel(BaseTransform):
    def __init__(self, max_size = 15) -> None:
        self.max_size = max_size

    def __call__(self, graph):
        mol = Chem.MolFromSmiles(graph.smiles)
        motif_counts = torch.zeros(self.max_size -2)

        rings = mol.GetRingInfo().AtomRings()
        for ring in rings:
            if len(ring) <= self.max_size:
                motif_counts[len(ring) - 3] += 1
        
        node_counts = torch.sum(graph.x, dim = 0)
        graph.motif_counts = torch.unsqueeze(torch.concat([motif_counts, node_counts], dim = 0), dim = 0)
        return graph
    
class PrincipalSubgraphNodeFeature(BaseTransform):
    def __init__(self, vocab: str) -> None:
        self.vocab = vocab
        self.tokenizer = Tokenizer(self.vocab)
        self.vocab_size = len(self.tokenizer.idx2subgraph) - 2 #contains two special symbols
    
    def __call__(self, graph):
        subgraph_mol = self.tokenizer(graph.smiles)
        node_features = torch.zeros(graph.num_nodes, self.vocab_size)
        for subgraph_node_id in subgraph_mol.nodes:
            atom_mapping =  subgraph_mol.get_node(subgraph_node_id).get_atom_mapping()
            atom_ids = list(atom_mapping.keys())
            subgraph_id = self.tokenizer.subgraph2idx[subgraph_mol.get_node(subgraph_node_id).smiles]
            node_features[atom_ids, subgraph_id] = 1
        graph.x = torch.cat([graph.x, node_features], dim = 1)
        return graph

class PrincipalSubgraphGraphLevel(BaseTransform):
    def __init__(self, vocab: str) -> None:
        self.vocab = vocab
        self.tokenizer = Tokenizer(self.vocab)
        self.vocab_size = len(self.tokenizer.idx2subgraph) - 2 #contains two special symbols
    
    def __call__(self, graph):
        subgraph_mol = self.tokenizer(graph.smiles)
        motif_counts = torch.zeros(self.vocab_size)
        for subgraph_node_id in subgraph_mol.nodes:
            subgraph_id = self.tokenizer.subgraph2idx[subgraph_mol.get_node(subgraph_node_id).smiles]
            motif_counts[subgraph_id] += 1
        node_counts = torch.sum(graph.x, dim = 0)
        graph.motif_counts = torch.unsqueeze(torch.concat([motif_counts, node_counts], dim = 0), dim = 0)
        return graph