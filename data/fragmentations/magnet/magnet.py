import warnings
from copy import deepcopy
from itertools import chain
from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import rdkit.DataStructs as DataStructs
import torch
from rdkit.Chem.rdchem import Mol

""" Adapted from https://github.com/TUM-DAML/MAGNet"""

ATOM_LIST = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "B", "Cu", "Zn", 'Co', "Mn", 'As', 'Al', 'Ni', 'Se', 'Si', 'H', 'He', 'Li', 'Be', 'Ne', 'Na', 'Mg', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Fe', 'Ga', 'Ge', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
             'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut', 'Fl', 'Uup', 'Lv', 'Uus', 'Uuo']


def compute_fingerprint(input: Union[str, Chem.rdchem.Mol]) -> np.array:
    if isinstance(input, str):
        mol = Chem.MolFromSmiles(input)
    else:
        mol = deepcopy(input)
    top_feats = np.packbits(Chem.RDKFingerprint(mol, fpSize=2048)) / 255
    circ_feats = AllChem.GetHashedMorganFingerprint(mol, 3, nBits=256)
    array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(circ_feats, array)
    circ_feats = array
    mol_fingerprint = np.concatenate([top_feats, circ_feats])
    return mol_fingerprint


def extract_valid_fragment(mol, extract_atom_ids):
    editeable_mol = Chem.RWMol()
    for i, eai in enumerate(extract_atom_ids):
        editeable_mol.AddAtom(Chem.Atom(mol.GetAtomWithIdx(eai).GetSymbol()))
        atom = editeable_mol.GetAtomWithIdx(i)
        atom.SetFormalCharge(mol.GetAtomWithIdx(eai).GetFormalCharge())
    for bond in mol.GetBonds():
        if bond.GetBeginAtom().GetIdx() in extract_atom_ids:
            if bond.GetEndAtom().GetIdx() in extract_atom_ids:
                ba = bond.GetBeginAtom().GetIdx()
                ba = extract_atom_ids.index(ba)
                ea = bond.GetEndAtom().GetIdx()
                ea = extract_atom_ids.index(ea)
                editeable_mol.AddBond(ba, ea, bond.GetBondType())
    return editeable_mol.GetMol()


def extract_fragment_from_mol(mol, extract_atom_ids):
    bonds_to_cut = []
    for bond in mol.GetBonds():
        atom_begin = bond.GetBeginAtom().GetIdx()
        atom_end = bond.GetEndAtom().GetIdx()
        if (atom_begin in extract_atom_ids) ^ (atom_end in extract_atom_ids):
            bonds_to_cut.append(bond.GetIdx())
    fragmented_molecule = Chem.FragmentOnBonds(
        mol, bonds_to_cut, addDummies=False)
    frag_idx = []
    frags = Chem.GetMolFrags(
        fragmented_molecule,
        asMols=True,
        sanitizeFrags=False,
        fragsMolAtomMapping=frag_idx,
    )
    for idx, frag in zip(frag_idx, frags):
        # result_mol = None
        if set(list(idx)) == set(extract_atom_ids):
            return idx, frag
        # if set(list(idx)).issubset(set(extract_atom_ids)):
        #     if not result_mol:
        #         result_mol = frag
        #     else:
        #         Chem.CombineMols(result_mol, frag)
    # return extract_atom_ids, frag
    warnings.warn("No Matching found")


class MolDecomposition:
    def __init__(self, input_mol: Union[str, Mol]):
        if isinstance(input_mol, str):
            mol = Chem.MolFromSmiles(input_mol)
            Chem.Kekulize(mol)
            self.mol = mol
        else:
            self.mol = input_mol
            Chem.Kekulize(self.mol)

        # initialize node mapping
        self.nodes = dict()
        for i in range(self.mol.GetNumAtoms()):
            self.nodes[i] = []

        # apply decomposition to molecule input
        self.set_leaf_atoms()
        self.decompose()
        # check: hypernode can only be in 2 motifs
        for v in self.nodes.values():
            assert 1 <= len(v) <= 2
        self.create_motif_map()
        # check: overlap of motifs can only be 1 node
        for key1 in self.id_to_hash.keys():
            shape_node_outer = [
                k for (k, v) in self.nodes.items() if key1 in v]
            for key2 in self.id_to_hash.keys():
                shape_node_inner = [
                    k for (k, v) in self.nodes.items() if key2 in v]
                if key1 != key2:
                    assert len(set(shape_node_outer).intersection(
                        set(shape_node_inner))) in [0, 1]
        # prepare features for __getitem__ call later, that is, the full decomposition including varying
        # granularities of mapping as well as other expensive featurization functions such as fingerprints
        self.prepare_fingerprints()
        self.prepare_batch_output()

    def prepare_fingerprints(self):
        self.fingerprint_mol = compute_fingerprint(self.mol)
        self.fingerprint_mol = np.array(self.fingerprint_mol, dtype=np.float32)
        if np.any(np.isnan(self.fingerprint_mol)):
            self.fingerprint_mol[np.isnan(self.fingerprint_mol)] = 0

    def decompose(self):
        core_mol_idx = [k for k, v in self.nodes.items() if len(v) == 0]
        # if molecule has no leaf atoms
        if len(core_mol_idx) == self.mol.GetNumAtoms():
            self.core_mol = self.mol
            idx_in_mol = tuple(core_mol_idx)
            self.valid_core_mol = self.mol
        else:
            # extract core molecule, i.e. without leafs
            idx_in_mol, self.core_mol = extract_fragment_from_mol(
                self.mol, [k for k, v in self.nodes.items() if len(v) == 0]
            )
            # core_mol preserves index ordering and might not encode a valid smiles
            # valid_core_mol is a valid molecule representing the core_mol
            self.valid_core_mol = extract_valid_fragment(
                self.mol, [k for k, v in self.nodes.items() if len(v) == 0]
            )
        # apply bbb to seperate cyclic bonds
        frag_idx = self.bbb_decomposition()

        # separate rings that are attached on only one joint
        frag_idx = self.decompose_rings(frag_idx)

        # junction decomposition
        frag_idx = self.junction_decomposition(frag_idx)

        # do actual motif assignment of nodes
        for k, idx in enumerate(frag_idx):
            for idx_in_core in idx:
                original_idx = idx_in_mol[idx_in_core]
                self.nodes[original_idx] = self.nodes[original_idx] + [k]

        # identify those bonds that are across motifs and make them separate motif
        for bond in self.mol.GetBonds():
            atom_begin = bond.GetBeginAtom().GetIdx()
            atom_end = bond.GetEndAtom().GetIdx()
            if self.nodes[atom_begin] == [-1] or self.nodes[atom_end] == [-1]:
                continue
            common_motif = set(self.nodes[atom_begin]).intersection(
                self.nodes[atom_end])
            # assert len(common_motif) <= 1
            if not common_motif:
                idx_in_mol, _ = extract_fragment_from_mol(
                    self.mol, sorted([atom_begin, atom_end]))
                current_class = max([max(v) for v in self.nodes.values()]) + 1
                self.nodes[atom_begin] = self.nodes[atom_begin] + \
                    [current_class]
                self.nodes[atom_end] = self.nodes[atom_end] + [current_class]

    def decompose_rings(self, frag_idx):
        frag_idx = list(frag_idx)
        while True:
            no_more_detaches_found = True
            for idx_in_core in frag_idx:
                # extract current fragment and find all junction atoms in fragment
                fragment = extract_valid_fragment(self.core_mol, idx_in_core)
                # if it is a cyclic structure, it can not have a junction
                if any([b.IsInRing() for b in fragment.GetBonds()]):
                    # check whether it is a single junction
                    for atom in fragment.GetAtoms():
                        if atom.GetDegree() == 4:
                            # check whether it connects two rings
                            ri = fragment.GetRingInfo()
                            neighbors = [n.GetIdx()
                                         for n in atom.GetNeighbors()]
                            check_ring_connector = [
                                [(t, n) for t in neighbors if (
                                    ri.AreAtomsInSameRing(t, n) and t != n)]
                                for n in neighbors
                            ]
                            # out of all neighbors, we expect exactly two to be in the same ring
                            if all([len(cr) == 1 for cr in check_ring_connector]):
                                if self.core_mol.GetAtomWithIdx(idx_in_core[atom.GetIdx()]).GetDegree() == 5:
                                    print(
                                        "WARNING: junction between two rings has another thing attached")
                                    continue
                                # pick any two neighbors and detach from them, add back connector atom later
                                no_more_detaches_found = False
                                cut_bonds = [
                                    fragment.GetBondBetweenAtoms(
                                        i, atom.GetIdx()).GetIdx()
                                    for i in check_ring_connector[0][0]
                                ]
                                core_mol_frags = Chem.FragmentOnBonds(
                                    fragment, cut_bonds, addDummies=False)
                                new_fragment_idx = Chem.GetMolFrags(
                                    core_mol_frags)
                                frag_idx.remove(idx_in_core)
                                for nfi in new_fragment_idx:
                                    if not atom.GetIdx() in nfi:
                                        nfi = list(nfi) + [atom.GetIdx()]
                                    frag_idx.append(
                                        tuple([idx_in_core[f] for f in nfi]))
                                break
            if no_more_detaches_found:
                break
        return frag_idx

    def set_leaf_atoms(self):
        adj = Chem.rdmolops.GetAdjacencyMatrix(self.mol)
        graph_no_leaf = nx.from_numpy_array(
            np.triu(adj), create_using=nx.Graph)
        for atom in self.mol.GetAtoms():
            graph_no_leaf.nodes[atom.GetIdx()]["label"] = atom.GetSymbol()
        atom_types, leaf_atoms = [], []
        for k in range(graph_no_leaf.number_of_nodes()):
            atom_types.append(ATOM_LIST.index(graph_no_leaf.nodes[k]["label"]))
        sorted_idx = np.flip(np.argsort(atom_types))
        for idx in sorted_idx:
            if graph_no_leaf.degree[idx.item()] == 1:
                neighbour = list(graph_no_leaf.neighbors(idx.item()))[0]
                neighbour_atom = self.mol.GetAtomWithIdx(neighbour)
                if graph_no_leaf.degree[neighbour] not in [2, 4]:
                    graph_no_leaf.remove_node(idx.item())
                    leaf_atoms.append(idx.item())
                elif neighbour_atom.IsInRing():
                    if neighbour_atom.GetDegree() == 4:
                        # get all neighbors of leaf atom neighbor
                        ri = self.mol.GetRingInfo()
                        nn = [n.GetIdx()
                              for n in neighbour_atom.GetNeighbors()]
                        nn = [n for n in nn if not ri.AreAtomsInSameRing(
                            n, neighbour)]
                        nn = [n for n in nn if n != idx]
                        # by definition, this ring should have 2 non-ring neighbors
                        # one of them is the leaf we are currently at
                        if len(nn) != 1:
                            graph_no_leaf.remove_node(idx.item())
                            leaf_atoms.append(idx.item())
                            continue
                        second_leaf_at_ring = self.mol.GetAtomWithIdx(nn[0])
                        # if the other neighbor neighbor is not a leaf, we can cut
                        if second_leaf_at_ring.GetDegree() > 1:
                            graph_no_leaf.remove_node(idx.item())
                            leaf_atoms.append(idx.item())

        for idx in leaf_atoms:
            self.nodes[idx] = self.nodes[idx] + [-1]

    def junction_decomposition(self, frag_idx):
        while True:
            updated_frag_idx = []
            junction_found = False
            # iterate through all fragment and check for junctions
            for idx_in_core in frag_idx:
                # extract current fragment and find all junction atoms in fragment
                fragment = extract_valid_fragment(self.core_mol, idx_in_core)
                # if it is a cyclic structure, it can not have a junction
                if all([b.IsInRing() for b in fragment.GetBonds()]):
                    updated_frag_idx.append(idx_in_core)
                    continue

                fragment_atoms = [a for a in fragment.GetAtoms()]

                junction_atoms = [a.GetIdx()
                                  for a in fragment_atoms if is_atom_junction(a)]
                # if we have found one, we start the cutting procedure
                if junction_atoms:
                    # connected junctions should stay intact, so we search from any junction atom
                    # if there are neighbors also with degree >= 3
                    current_junction = [junction_atoms.pop(0)]
                    ri = fragment.GetRingInfo()
                    while True:
                        neighbor_found = False
                        for start_node in current_junction:
                            for n in fragment.GetAtomWithIdx(start_node).GetNeighbors():
                                if is_atom_junction(n):
                                    if n.GetIdx() not in current_junction:
                                        neighbor_found = True
                                        junction_atoms.remove(n.GetIdx())
                                        current_junction.append(n.GetIdx())
                        # if it is a cyclic junction we have to add the entire ring to the shape
                        while True:
                            len_before = len(current_junction)
                            new_junction_members = []
                            for jm in current_junction:
                                for a in fragment_atoms:
                                    if ri.AreAtomsInSameRing(jm, a.GetIdx()):
                                        new_junction_members.append(a.GetIdx())
                            current_junction = list(
                                set(current_junction + new_junction_members))
                            if len(current_junction) == len_before:
                                break
                            else:
                                neighbor_found = True
                        if not neighbor_found:
                            break

                    # find all neighbors of junction to add to fragment
                    junction_neighbors = []
                    for j in current_junction:
                        j_atom = fragment.GetAtomWithIdx(j)
                        if is_atom_junction(j_atom):
                            neighbors = j_atom.GetNeighbors()
                            junction_neighbors.extend(
                                [n.GetIdx() for n in neighbors])
                    junction_members = junction_neighbors + current_junction

                    # cut all bonds that go outside of fragment except
                    # those in rings because they constitute ring junctions
                    cut_bonds = []
                    for b in fragment.GetBonds():
                        ba, ea = b.GetBeginAtom().GetIdx(), b.GetEndAtom().GetIdx()
                        if (ba in junction_members) ^ (ea in junction_members):
                            assert not b.IsInRing()
                            cut_bonds.append(b.GetIdx())

                    if cut_bonds:
                        junction_found = True
                        core_mol_frags = Chem.FragmentOnBonds(
                            fragment, cut_bonds, addDummies=False)
                        frag_idx = Chem.GetMolFrags(core_mol_frags)
                        # assign idx double to create hypernodes
                        for f_idx in frag_idx:
                            # junction fragment encountered, we just update fragment idx
                            if any([f in junction_members for f in f_idx]):
                                updated_frag_idx.append(
                                    tuple([idx_in_core[f] for f in f_idx]))
                                continue
                            # for other, cut fragments, we need to determine hypernodes
                            f_idx = set(f_idx)
                            add_ids = []
                            for b_id in cut_bonds:
                                # if any of the cut bonds coindices with atom in other fragment
                                bond = fragment.GetBondWithIdx(b_id)
                                ba, ea = bond.GetBeginAtom(), bond.GetEndAtom()
                                bond_set = set([ba.GetIdx(), ea.GetIdx()])
                                shared_nodes = f_idx.intersection(bond_set)
                                # add end node of bond as hypernode
                                if shared_nodes:
                                    add_ids.append(
                                        list(bond_set - shared_nodes)[0])
                            f_idx = tuple(f_idx.union(set(add_ids)))
                            updated_frag_idx.append(
                                tuple([idx_in_core[f] for f in f_idx]))
                    else:
                        updated_frag_idx.append(idx_in_core)
                else:
                    updated_frag_idx.append(idx_in_core)
                    continue
            frag_idx = updated_frag_idx
            if not junction_found:
                break
        return updated_frag_idx

    def bbb_decomposition(self):
        ids_of_bonds_to_cut = []
        for bond in self.core_mol.GetBonds():
            if bond.IsInRing():
                continue
            atom_begin = bond.GetBeginAtom()
            atom_end = bond.GetEndAtom()
            if not atom_begin.IsInRing() and not atom_end.IsInRing():
                continue
            if atom_begin.IsInRing() and not atom_end.IsInRing():
                if is_atom_junction(atom_begin):
                    continue
            elif atom_end.IsInRing() and not atom_begin.IsInRing():
                if is_atom_junction(atom_end):
                    continue
            if is_atom_junction(atom_end) and is_atom_junction(atom_begin):
                continue
            ids_of_bonds_to_cut.append(bond.GetIdx())
        if ids_of_bonds_to_cut:
            core_mol_frags = Chem.FragmentOnBonds(
                self.core_mol, ids_of_bonds_to_cut, addDummies=False)
            frag_idx = Chem.GetMolFrags(core_mol_frags)
            is_ring_junction = []
            for f_idx in frag_idx:
                atoms_in_frag = [
                    self.core_mol.GetAtomWithIdx(f) for f in f_idx]
                is_junction = any([is_atom_junction(a) for a in atoms_in_frag])
                is_ring = any([a.IsInRing() for a in atoms_in_frag])
                is_ring_junction.append(is_junction and is_ring)
            # assign idx double to create hypernodes
            updated_frag_idx = []
            for j, f_idx in enumerate(frag_idx):
                # cyclic motifs already have all nodes they need
                fragment = extract_valid_fragment(self.core_mol, f_idx)
                # if it is a cyclic structure, it can not have a junction
                # we need to handle single atoms, all([]) = True!
                if all([b.IsInRing() for b in fragment.GetBonds()]) and (fragment.GetNumAtoms() > 1):
                    if not any([is_atom_junction(self.core_mol.GetAtomWithIdx(f)) for f in f_idx]):
                        updated_frag_idx.append(f_idx)
                        continue
                # chains and junctions need hypernodes to be added
                f_idx = set(f_idx)
                add_ids = []
                for b_id in ids_of_bonds_to_cut:
                    bond = self.core_mol.GetBondWithIdx(b_id)
                    ba, ea = bond.GetBeginAtom(), bond.GetEndAtom()
                    bond_set = set([ba.GetIdx(), ea.GetIdx()])
                    shared_nodes = f_idx.intersection(bond_set)
                    assert len(shared_nodes) <= 1
                    if shared_nodes:
                        shared_node_atom = self.core_mol.GetAtomWithIdx(
                            list(shared_nodes)[0])
                        # junction cycle special case
                        if self.core_mol.GetAtomWithIdx(list(shared_nodes)[0]).IsInRing():
                            if not is_atom_junction(shared_node_atom):
                                continue
                        add_ids.append(list(bond_set - shared_nodes)[0])
                f_idx = tuple(f_idx.union(set(add_ids)))
                updated_frag_idx.append(f_idx)
            return updated_frag_idx
        else:
            # if BBB does not want to cut any bond
            return (tuple(range(self.core_mol.GetNumAtoms())),)

    def create_motif_map(self):
        self.id_to_fragment, self.id_to_hash, self.hash_to_id = dict(), dict(), dict()
        self.id_to_hash[-1] = -1
        self.hash_to_id[-1] = -1
        num_classes = max(list(chain(*list(self.nodes.values()))))
        for i in range(num_classes + 1):
            atoms_in_motif = [k for k, v in self.nodes.items() if (i in v)]
            frag = extract_valid_fragment(self.mol, atoms_in_motif)
            # Attention: since we sanitze, we can not rely on the ordering of the smiles
            Chem.SanitizeMol(frag)
            adjacency = Chem.GetAdjacencyMatrix(frag)
            graph = nx.from_numpy_array(
                np.triu(adjacency), create_using=nx.Graph)
            graph_hash = nx.weisfeiler_lehman_graph_hash(graph)
            self.id_to_hash[i] = graph_hash
            self.id_to_fragment[i] = Chem.MolToSmiles(frag)
            self.hash_to_id[graph_hash] = i

    def create_hypergraph(self):
        num_classes = len(
            set([c for c in chain(*list(self.nodes.values())) if c != -1]))
        graph = nx.Graph()
        for i in range(num_classes):
            graph.add_node(i)
        # add edge automatically adds required nodes
        for class_assignment in self.nodes.values():
            if len(class_assignment) == 2:
                graph.add_edge(class_assignment[0], class_assignment[1])
        return graph

    def plot_decomposition(self):
        for i, atom in enumerate(self.mol.GetAtoms()):
            # For each atom, set the property "atomLabel" to a custom value, let's say a1, a2, a3,...
            atom.SetProp("atomLabel", f"{self.nodes[i]}")
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))
        ax1.imshow(Chem.Draw.MolToImage(self.mol))
        graph = self.create_hypergraph()
        nx.draw(graph, labels=self.id_to_fragment, ax=ax2)
        plt.show()

    def prepare_batch_output(self, mfeat_shape=(512,)):
        shape_nodes, gt_motifs, allowed_joins = [], [], dict()
        for key, _ in self.id_to_hash.items():
            if key == -1:
                continue
            shape_node_idx = [k for (k, v) in self.nodes.items() if key in v]
            # canonically sort shape node idx here s.t. at join inference,
            # when we use these ids to mask, they are already sorted canonically for pos. encoding
            motif = extract_valid_fragment(self.mol, shape_node_idx)
            _ = Chem.MolToSmiles(motif)
            s_idx = list(motif.GetPropsAsDict(includePrivate=True,
                         includeComputed=True)["_smilesAtomOutputOrder"])
            s_idx = np.argsort(s_idx)
            shape_node_idx = np.array(shape_node_idx)[s_idx]
            node_degrees = np.array([a.GetDegree()
                                    for a in motif.GetAtoms()])[s_idx]
            shape_nodes.append(shape_node_idx.tolist())
            # if it is cyclic all atoms can do joins
            if is_all_cyclic(motif):
                for sni in shape_node_idx:
                    allowed_joins[sni] = True
            # cyclic junctions
            elif is_cyclic_junction(motif):
                is_junc_atom = (~np.array([is_atom_cyclic_junction(a) for a in motif.GetAtoms()])).astype(int)[
                    s_idx
                ]
                for j, sni in enumerate(shape_node_idx):
                    allowed_joins[sni] = is_junc_atom[j]
            # chains or non-cyclic junctions
            else:
                for sni, nd in zip(shape_node_idx, node_degrees):
                    if sni in allowed_joins.keys():
                        assert allowed_joins[sni]
                        continue
                    allowed_joins[sni] = nd == 1
            gt_motif = self.id_to_fragment[key]
            gt_motif = Chem.MolToSmiles(Chem.MolFromSmiles(
                gt_motif), isomericSmiles=False, kekuleSmiles=True)
            gt_motifs.append(gt_motif)

        hgraph = self.create_hypergraph()
        # extract hashes and fill up with -1 in case its not a join, i.e. does not have a secondary shape
        shape_classes = [[self.id_to_hash[i]
                          for i in (v + [-1])[:2]] for v in self.nodes.values()]
        feats_per_motif = [compute_fingerprint(sm) for sm in gt_motifs]

        # map hash ids to previously computed features
        def map_hash_to_feat(hashes):
            return [
                np.full(mfeat_shape, fill_value=-1) if h == -
                1 else feats_per_motif[self.hash_to_id[h]]
                for h in hashes
            ]

        motif_features = [map_hash_to_feat(s) for s in shape_classes]
        motif_features = [np.concatenate(mfeat) for mfeat in motif_features]
        self.batch_out = dict(
            hgraph_nodes=[h for h in self.id_to_hash.values()],
            hgraph_adj=nx.to_numpy_array(hgraph),
            shape_classes=shape_classes,
            motif_features=np.stack(motif_features, axis=0),
            nodes_in_shape=shape_nodes,
            gt_motif_smiles=gt_motifs,
            allowed_joins=allowed_joins,
            feats_per_motif=feats_per_motif,
        )

    def get_batch_output(self, hash_to_class_map):
        hgraph_nodes = [hash_to_class_map[hn]
                        for hn in self.batch_out["hgraph_nodes"] if hn != -1]
        self.batch_out["hgraph_nodes"] = hgraph_nodes
        values, counts = np.unique(np.array(hgraph_nodes), return_counts=True)
        self.batch_out["shape_classes"] = [
            [hash_to_class_map[c1], c2] if c1 != -1 else [c1, c2] for c1, c2 in self.batch_out["shape_classes"]
        ]
        self.batch_out["shape_classes"] = [
            [c1, hash_to_class_map[c2]] if c2 != -1 else [c1, c2] for c1, c2 in self.batch_out["shape_classes"]
        ]
        self.batch_out["shape_classes"] = np.array(
            self.batch_out["shape_classes"])
        hash_mult = []
        multiplicity_per_class = np.zeros((len(hash_to_class_map.values()),))
        for _, hash in self.id_to_hash.items():
            if hash == -1:
                continue
            hid = hash_to_class_map[hash]
            hash_mult.append(multiplicity_per_class[hid].astype(int).item())
            multiplicity_per_class[hid] += 1
        assert counts.sum() == multiplicity_per_class.sum()
        self.batch_out["hgraph_nodes_mult"] = hash_mult
        return self.batch_out


def is_all_cyclic(mol):
    all_bonds = all([b.IsInRing() for b in mol.GetBonds()])
    all_atoms = all([a.IsInRing() for a in mol.GetAtoms()])
    return all_bonds and all_atoms


def is_cyclic_junction(mol):
    return any([is_atom_cyclic_junction(a) for a in mol.GetAtoms()])


def is_atom_cyclic_junction(atom):
    return atom.IsInRing() and is_atom_junction(atom)


def is_atom_junction(atom):
    if atom.IsInRing():
        if (sum([0 if n.IsInRing() else 1 for n in atom.GetBonds()]) == 2) and (atom.GetDegree() == 4):
            return True
        else:
            return False
    else:
        return atom.GetDegree() in [3, 4]
