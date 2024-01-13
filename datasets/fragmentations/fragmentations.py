from datasets.fragmentations.magnet import MolDecomposition
from datasets.fragmentations.breaking_bridge_bonds import find_motifs_from_vocabulary, MotifExtractionSettings, MotifVocabularyExtractor
from datasets.fragmentations.mol_bpe import Tokenizer, graph_bpe_smiles
from collections import Counter
from rdkit.Chem.BRICS import BRICSDecompose, BreakBRICSBonds
from rdkit import Chem
from typing import Any, List, Literal
from rdkit.Chem.rdmolops import GetMolFrags
from torch_geometric.transforms import BaseTransform
import torch
from torch_geometric.data import Data
import pickle
from itertools import permutations
import numpy as np

fragment2type = {"ring": 0 , "path": 1, "junction": 2}
ATOM_LIST = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Cu", "Zn", 'Co', "Mn", 'As', 'Al', 'Ni', 'Se', 'Si', 'H', 'He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Fe', 'Ga', 'Ge', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut', 'Fl', 'Uup', 'Lv', 'Uus', 'Uuo']

def is_leaf(node_id, graph):

    neighbors = get_neighbors(node_id, graph)
    if len(neighbors) == 1:
        neighbor = neighbors[0]
        if graph.mol.GetAtomWithIdx(neighbor).IsInRing():
            return True
        nns = get_neighbors(neighbor, graph)
        degree_nn = [get_degree(nn, graph) for nn in nns]
        if len([degree for degree in degree_nn if degree >= 2]) >= 2:
            return True
        # one neighbor neighbor with degree one is not a leaf
        potential_leafs = [nn for nn in nns if get_degree(nn, graph) == 1]
        atom_types = [(ATOM_LIST.index(graph.mol.GetAtomWithIdx(nn).GetSymbol()), nn) for nn in potential_leafs]
        sorted_idx = np.sort(atom_types)
        if sorted_idx[-1][1] == node_id:
            #node at end of path
            return False
        else:
            return True
    return False
        





def vocab_to_file(vocab, file_name):
    if not file_name:
        file_name = f"./{vocab.__name__}_vocab_{vocab.max_vocab_size}"
    pickle.dump(vocab.get_vocab(), open(file_name, "wb"))

def get_vocab_from_file(file_name):
    return pickle.load(open(file_name, "rb"))

def get_neighbors(node_id, graph):
    return (graph.edge_index[1, graph.edge_index[0,:] == node_id]).tolist()

def get_degree(node_id, graph):
    return len(get_neighbors(node_id, graph))

class BreakingBridgeBondsVocab(BaseTransform):
    def __init__(self, min_frequency = None, min_num_atoms = 3, cut_leaf_edges = False, vocab_size = 200):
        settings = MotifExtractionSettings(min_frequency = min_frequency, min_num_atoms=min_num_atoms, cut_leaf_edges=cut_leaf_edges, max_vocab_size=vocab_size)
        self.extractor = MotifVocabularyExtractor(settings)
        self.max_vocab_size = vocab_size
    
    def __call__(self, graph):
        mol = graph.mol
        self.extractor.update(mol)
        return graph
    
    def get_vocab(self):
        return self.extractor.output()
    

class BRICSVocab(BaseTransform):
    def __init__(self, vocab_size = 200):
        self.max_vocab_size = vocab_size
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
    def __init__(self, vocab_size = 200, vocab_path = "./principal_subgraph_vocab.txt", cpus = 4, kekulize = False):
        self.max_vocab_size = vocab_size
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
    def __init__(self, vocab_size = 200):
        self.max_vocab_size = vocab_size
        self.hash_counter = Counter()
    
    def __call__(self, graph):
        mols = Chem.Mol(graph.mol) #create copy of molecule
        for mol in Chem.rdmolops.GetMolFrags(mols, asMols = True):
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
        mols = Chem.Mol(graph.mol) #create copy of molecule
        for mol in Chem.rdmolops.GetMolFrags(mols, asMols = True):
            #There can be multiple disconnected parts of a molecule
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

class MagnetWithoutVocab(BaseTransform):
    def __init__(self, vocab_size = None):
        pass

    def __call__(self, graph):
        mols = Chem.Mol(graph.mol) #create copy of molecule
        fragment_types = []
        node_substructures = []
        for mol in Chem.rdmolops.GetMolFrags(mols, asMols = True):
            #There can be multiple disconnected parts of a molecule
            decomposition = MolDecomposition(mol)
                     
            fragment_to_index = {}
            fragment_to_type = {}
            for fragments in decomposition.nodes.values():
                fragment_info = []
                for frag_id in fragments:
                    if frag_id == -1:
                        #don't use leafs ?!
                        continue

                    if frag_id not in fragment_to_type:
                        frag_mol = Chem.MolFromSmiles(decomposition.id_to_fragment[frag_id])
                        if frag_mol.GetAtomWithIdx(0).IsInRing():
                            fragment_to_type[frag_id] = [fragment2type["ring"], frag_mol.GetNumAtoms()]
                        elif all([a.GetDegree() in [1, 2] for a in frag_mol.GetAtoms()]):
                            fragment_to_type[frag_id] = [fragment2type["path"],frag_mol.GetNumAtoms()]
                        else:
                            fragment_to_type[frag_id] = [fragment2type["junction"], frag_mol.GetNumAtoms()]

                    if frag_id not in fragment_to_index:
                        fragment_to_index[frag_id] = len(fragment_types)
                        fragment_types.append(fragment_to_type[frag_id])

                    fragment_info.append((fragment_to_index[frag_id], fragment_to_type[frag_id][0]))

                node_substructures.append(fragment_info)
        
        if fragment_types:
            graph.fragment_types = torch.tensor(fragment_types, dtype = torch.long)
        else:
            graph.fragment_types = torch.empty((0,2), dtype = torch.long)
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
    def __init__(self, vocab_size = 15) -> None:
        """Initialize ring based fragmentation that finds a set of short Rings 
        covering all rings of the molecule.

        Parameters
        ----------
        max_vocab_size, optional
            Maximum vocab size, i.e. size of the longest ring - 2, by default 15
        """
        self.max_vocab_size = vocab_size
        self.max_ring_size = vocab_size + 2
    
    def __call__(self, graph):
        mol = graph.mol
        rings = Chem.GetSymmSSSR(mol)
        node_substructures = [[] for _ in range(graph.num_nodes)]
        fragment_types = []
        fragment_id = 0
        for i in range(len(rings)):
            ring = list(rings[i])
            fragment_types.append([fragment2type["ring"], len(ring)])
            if len(ring) <= self.max_ring_size:
                for atom in ring:
                    fragment_type = len(ring) - 3
                    node_substructures[atom].append((fragment_id, fragment_type))
                fragment_id += 1
            else:
                for atom in ring:
                    fragment_type = self.max_vocab_size - 1 # max fragment_type number
                    node_substructures[atom].append((fragment_id, fragment_type))
                fragment_id += 1
        graph.substructures = node_substructures
        if fragment_types:
            graph.fragment_types = torch.tensor(fragment_types, dtype = torch.long)
        else:
            graph.fragment_types = torch.empty((0,2), dtype = torch.long)
        return graph

class RingsEdges(BaseTransform):
    def __init__(self, vocab_size, cut_leafs = False):
        self.max_ring = vocab_size - 1 # one fragment for edges
        self.rings = Rings(self.max_ring)
        self.cut_leafs = cut_leafs
    
    def __call__(self, graph):
        self.rings(graph)

        #now find edges not in rings
        max_frag_id = max([frag_id for frag_infos in graph.substructures for (frag_id, _) in frag_infos], default = -1)
        fragment_id = max_frag_id + 1

        fragment_types = []

        for bond in graph.mol.GetBonds():
            if not bond.IsInRing():
                #add bond as new fragment
                atom1 = bond.GetBeginAtomIdx()
                atom2 = bond.GetEndAtomIdx()
                if self.cut_leafs and (is_leaf(atom1, graph) or is_leaf(atom2, graph)):
                    continue
                fragment_types.append([fragment2type["path"],2])
                bond_info = (fragment_id, self.max_ring)
                fragment_id += 1
                graph.substructures[atom1].append(bond_info)
                graph.substructures[atom2].append(bond_info)

        graph.fragment_types = torch.concat([graph.fragment_types, torch.tensor(fragment_types, dtype = torch.long)], dim = 0)
        return graph


class RingsPaths(BaseTransform):
    def __init__(self, vocab_size = 30, max_ring = 15, cut_leafs = False):
        self.max_ring = max_ring
        assert(vocab_size > max_ring)
        self.max_path = vocab_size - max_ring
        self.rings = Rings(max_ring)
        self.cut_leafs = cut_leafs
    
    def get_frag_type(self, type: Literal["ring", "path"], size):
        if type == "ring":
            return size - 3 if size - 3  < self.max_ring else self.max_ring - 1
        else: # type == "path"
            offset = self.max_ring
            return  offset + size - 2 if size - 2 < self.max_path else offset + self.max_path - 1

    def __call__(self, graph):
        #first find rings
        self.rings(graph)

        #now find paths
        max_frag_id = max([frag_id for frag_infos in graph.substructures for (frag_id, _) in frag_infos], default = -1)
        fragment_id = max_frag_id + 1

        fragment_types = []

        #find paths
        visited = set()
        for bond in graph.mol.GetBonds():
            
            if not bond.IsInRing() and bond.GetIdx() not in visited:
                if self.cut_leafs and is_leaf(bond.GetBeginAtomIdx(), graph) and is_leaf(bond.GetEndAtomIdx(), graph):
                    continue
                visited.add(bond.GetIdx())
                in_path = []
                to_do = set([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
                while to_do:
                    next_node = to_do.pop()
                    in_path.append(next_node)
                    neighbors = [neighbor for neighbor in get_neighbors(next_node, graph) if not is_leaf(neighbor, graph) or not self.cut_leafs]
                    if not graph.mol.GetAtomWithIdx(next_node).IsInRing() and not len(neighbors) > 2:
                        #not in ring and not a junction
                        new_neighbors = [neighbor for neighbor in neighbors if neighbor not in in_path]
                        visited.update([graph.mol.GetBondBetweenAtoms(next_node, neighbor).GetIdx() for neighbor in new_neighbors])
                        to_do.update(new_neighbors)
                
                path_info = (fragment_id, self.get_frag_type("path", len(in_path)))
                fragment_types.append([fragment2type["path"], len(in_path)])
                fragment_id += 1
                for node_id in in_path:
                    graph.substructures[node_id].append(path_info)

        graph.fragment_types = torch.concat([graph.fragment_types, torch.tensor(fragment_types, dtype = torch.long)], dim = 0)
        # #find junctions
        # for node_id in range(graph.num_nodes):
        #     if not graph.mol.GetAtomWithIdx(node_id).IsInRing():
        #         neighbors = get_neighbors(node_id, graph)
        #         if len(neighbors) > 2:
        #             graph.substructures[node_id].append((fragment_id, self.get_frag_type("junction", len(neighbors))))
        #             fragment_id += 1
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
        elif key == "higher_edge_index":
            return self.fragments.size(0)
        elif key == "low_high_edge_index":
            return torch.tensor([[self.edge_index.size(1)], [self.fragments.size(0)]])
        elif key == "join_node_index":
            return torch.tensor([[self.higher_edge_index.size(1)], [self.x.size(0)]])
        elif key == "join_edge_index":
            return torch.tensor([[self.higher_edge_index.size(1)], [self.edge_index.size(1)]])
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
        if not edges:
            graph.fragments_edge_index = torch.empty((2,0), dtype = torch.long)
        else:
            graph.fragments_edge_index = torch.tensor(edges, dtype = torch.long).T.contiguous()
        
        # get (low level) edges that are part of a fragment
        low_high_edges = []
        for edge_id, (node_a, node_b) in enumerate(graph.edge_index.T):
            overlapping_fragments = set(graph.substructures[node_a]).intersection(set(graph.substructures[node_b]))
            for (frag_id, _) in overlapping_fragments:
                low_high_edges.append([edge_id, frag_id])
        if not low_high_edges:
            graph.low_high_edge_index = torch.empty((2,0), dtype = torch.long)
        else:
            graph.low_high_edge_index = torch.tensor(low_high_edges, dtype = torch.long).T.contiguous()

        return FragmentData(**{k: v for k, v in graph})
    
class HigherLevelGraph(BaseTransform):
    def __init__(self, vocab_size, neighbor: Literal["node", "tree"],higher_edge_features = False):
        self.vocab_size = vocab_size
        self.neighbor = neighbor
        self.frag_rep = FragmentRepresentation(vocab_size) if neighbor != "tree" else FragmentRepresentation(vocab_size + 1)
        self.higher_edge_features = higher_edge_features
    
    def __call__(self, graph):
        higher_edges = []
        if self.neighbor == "node":
            for frag_infos in graph.substructures:
                frag_ids = [frag_id for frag_id, _ in frag_infos]
                if len(frag_ids) >= 2:
                    #node in at least two substructures
                    for frag1, frag2 in permutations(frag_ids, 2):
                        #add substructure edge
                        if [frag1, frag2] not in higher_edges:
                            higher_edges.append([frag1, frag2])
        elif self.neighbor == "tree":
            fragment_types = []
            max_frag_id = max([frag_id for frag_infos in graph.substructures for (frag_id, _) in frag_infos], default = -1)
            fragment_id = max_frag_id + 1

            for node_id, frag_infos in enumerate(graph.substructures):
                frag_ids = [frag_id for frag_id, _ in frag_infos]
                if len(frag_ids) == 2:
                    #node in exactly two substructures
                    for frag1, frag2 in permutations(frag_ids, 2):
                        #add substructure edge
                        if [frag1, frag2] not in higher_edges:
                            higher_edges.append([frag1, frag2])
                elif len(frag_ids) > 2:
                    #node in more than two substructures, introduce junction
                    junction_id = fragment_id
                    graph.substructures[node_id].append((junction_id, self.vocab_size))
                    fragment_types.append([fragment2type["junction"], len(frag_ids)])
                    fragment_id += 1
                    for frag_id in frag_ids:
                        #add substructure edge
                        higher_edges.append([frag_id, junction_id])
                        higher_edges.append([junction_id, frag_id])
            if hasattr(graph, "fragment_types"):
                graph.fragment_types = torch.concat([graph.fragment_types, torch.tensor(fragment_types, dtype = torch.long)], dim = 0)
        else:
            raise RuntimeError("Unsupported neighborhood option")
        
        graph = self.frag_rep(graph)
        
        if not higher_edges:
            graph.higher_edge_index = torch.empty((2,0), dtype = torch.long)
        else:
            graph.higher_edge_index = torch.tensor(higher_edges, dtype = torch.long).T.contiguous()
        
        if self.higher_edge_features:
            #compute join nodes/edges for higher level graph
            join_nodes_list = []
            join_edges_list = []
            edge_types_list = []
            for higher_edge_id, higher_edge in enumerate(graph.higher_edge_index.T):
                nodes1 = graph.fragments_edge_index[0, graph.fragments_edge_index[1,:] == higher_edge[0]]
                nodes2 = graph.fragments_edge_index[0, graph.fragments_edge_index[1,:] == higher_edge[1]]
                join_nodes = [node1 for node1 in nodes1 if node1 in nodes2]
                join_edges = [edge_id for edge_id, (node_a, node_b) in enumerate(graph.edge_index.T) if node_a in join_nodes and node_b in join_nodes]
                join_nodes_list += [[higher_edge_id, join_node] for join_node in join_nodes]
                join_edges_list += [[higher_edge_id, join_edge] for join_edge in join_edges]
                if join_edges:
                    edge_types_list.append(1)
                else:
                    edge_types_list.append(0)

            if not join_nodes_list:
                graph.join_node_index = torch.empty((2,0), dtype = torch.long)
            else:
                graph.join_node_index = torch.tensor(join_nodes_list, dtype = torch.long).T.contiguous()
            if not join_edges_list:
                graph.join_edge_index = torch.empty((2,0), dtype = torch.long)
            else:
                graph.join_edge_index = torch.tensor(join_edges_list, dtype = torch.long).T.contiguous()
            if not edge_types_list:
                graph.higher_edge_types = torch.empty((0), dtype = torch.long)
            else:
                graph.higher_edge_types = torch.tensor(edge_types_list, dtype = torch.long)

        return graph
        


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