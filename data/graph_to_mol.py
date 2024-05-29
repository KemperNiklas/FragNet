"""Adapted from https://github.com/TUM-DAML/synthetic_coordinates/"""
import pandas as pd
import torch
# from rdkit.Chem import AllChem as Chem
from rdkit import Chem
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdmolops import FastFindRings
from torch_geometric.transforms import BaseTransform
from torch_scatter import scatter

# simple atoms without charge
# map ndx to atomic number
NODE_MAPPING1 = {
    0: 6,  # C
    1: 8,  # O
    2: 7,  # N
    3: 9,  # F
    5: 16,  # S
    6: 17,  # Cl
    9: 35,  # Br
    15: 53,  # I
    16: 15,  # P
}

# single atoms with charge
# ndx -> (atomic number, charge)
NODE_MAPPING2 = {
    7: (8, -1),  # O-
    12: (7, 1),  # N+
    13: (7, -1),  # N-
    14: (16, -1),  # S-
    19: (8, 1),  # O+
    20: (16, 1),  # S+
    24: (15, 1),  # P+
}

# radicals/groups with more than 1 atom (heavy atom + hydrogen)
# ndx -> (atomic number, numH, charge)
NODE_MAPPING3 = {
    4: (6, 0, 1),  # CH1
    8: (7, 1, 1),  # NH1+
    10: (7, 3, 1),  # NH3+
    11: (7, 2, 1),  # NH2+
    17: (7, 1, 1),  # OH1+
    18: (7, 1, 1),  # NH1+
    21: (15, 1, 1),  # PH1
    22: (15, 2, 0),  # PH2
    23: (6, 2, -1),  # CH2-
    25: (16, 1, 1),  # SH1+
    26: (6, 1, -1),  # CH1-
    27: (15, 1, 1),  # PH1+
}


def zinc_bond_ndx_to_bond(ndx):
    mapping = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }
    return mapping[ndx]


def atom_ndx_to_atom(ndx):
    """
    Create the Chem.Atom corresponding to ndx
    """
    if ndx in NODE_MAPPING1:
        return Chem.Atom(NODE_MAPPING1[ndx])

    if ndx in NODE_MAPPING2:
        atom_num, charge = NODE_MAPPING2[ndx]
        atom = Chem.Atom(atom_num)
        atom.SetFormalCharge(charge)
        return atom

    if ndx in NODE_MAPPING3:
        atom_num, _, charge = NODE_MAPPING3[ndx]
        atom = Chem.Atom(atom_num)
        atom.SetFormalCharge(charge)
        return atom

    raise ValueError


class ZINC_Graph_Add_Mol(BaseTransform):
    """
    Add rdkit mol object to ZINC graph. 
    In some edge cases the molecule night not exactly correspond to the original molecule.
    """

    def __init__(self):
        pass

    def __call__(self, graph):
        """
        Map node labels:
        """
        # create a read/write molecule
        mol = Chem.RWMol()

        add_hs = {}

        # convert to atomic number value
        atom_nums = graph.x.squeeze().numpy().tolist()

        for atom_ndx, ndx in enumerate(atom_nums):
            # add an atom with this atomic number
            mol.AddAtom(atom_ndx_to_atom(ndx))

            # check if its heavy atom+H
            if ndx in NODE_MAPPING3:
                _, num_h, _ = NODE_MAPPING3[ndx]
                # store the number of Hs to be added
                add_hs[atom_ndx] = num_h

        # # where to start indexing new H atoms
        # h_ndx = len(graph.x)

        # for atom_ndx, num_hs in add_hs.items():
        #     for _ in range(num_hs):
        #         mol.AddAtom(Chem.Atom(1))
        #         # bond from atom to this H
        #         mol.AddBond(atom_ndx, h_ndx, Chem.BondType.SINGLE)
        #         # increment every time a H is added
        #         h_ndx += 1

        """
        Map Edge labels:
        'SINGLE': 1
        'DOUBLE': 2
        'TRIPLE': 3
        """
        # bond type for each bond - single, double, triple
        bond_vals = graph.edge_attr.squeeze().numpy()
        bonds = list(map(zinc_bond_ndx_to_bond, bond_vals))

        # create a bond, set its properties and add to the molecule
        for ndx, (i, j) in enumerate(graph.edge_index.T):
            bond = mol.GetBondBetweenAtoms(i.item(), j.item())
            if bond is None:
                mol.AddBond(i.item(), j.item(), bonds[ndx])

        # cleanup the molecule
        Chem.SanitizeMol(mol)
        mol.UpdatePropertyCache()
        # FastFindRings(mol)

        # smi = Chem.MolToSmiles(mol)
        graph.mol = mol
        return graph


def get_chiral_tag(ndx):
    mapping = {
        0: Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        1: Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        2: Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        3: Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        4: Chem.rdchem.ChiralType.CHI_OTHER,
    }
    return mapping[ndx]


def get_formal_charge(ndx):
    charges = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    mapping = dict(zip(range(len(charges)), charges))
    return mapping[ndx]


def get_hybridization(ndx):
    mapping = {
        0: Chem.rdchem.HybridizationType.SP,
        1: Chem.rdchem.HybridizationType.SP2,
        2: Chem.rdchem.HybridizationType.SP3,
        3: Chem.rdchem.HybridizationType.SP3D,
        4: Chem.rdchem.HybridizationType.SP3D2,
        5: Chem.rdchem.HybridizationType.OTHER,
    }
    return mapping[ndx]


def get_aromatic(ndx):
    mapping = {0: False, 1: True}
    return mapping[ndx]


def bond_ndx_to_bond(ndx):
    mapping = {
        0: Chem.BondType.SINGLE,
        1: Chem.BondType.DOUBLE,
        2: Chem.BondType.TRIPLE,
        3: Chem.BondType.AROMATIC,
    }
    return mapping[ndx]


def get_bond_stereo(ndx):
    mapping = {
        0: Chem.rdchem.BondStereo.STEREONONE,
        1: Chem.rdchem.BondStereo.STEREOZ,
        2: Chem.rdchem.BondStereo.STEREOE,
        3: Chem.rdchem.BondStereo.STEREOCIS,
        4: Chem.rdchem.BondStereo.STEREOTRANS,
        5: Chem.rdchem.BondStereo.STEREOANY,
    }
    return mapping[ndx]


def get_conjugated(ndx):
    mapping = {0: False, 1: True}
    return mapping[ndx]


def extract_node_feature(data, reduce="add"):
    if reduce in ["mean", "max", "add"]:
        data.x = scatter(
            data.edge_attr,
            data.edge_index[0],
            dim=0,
            dim_size=data.num_nodes,
            reduce=reduce,
        )
    else:
        raise Exception("Unknown Aggregation Type")
    return data


class OGB_Graph_Add_Mol_By_Smiles:
    """
    Add rdkit mol object to OGB hiv graph
    """

    def __init__(self, filename='datasets/data/ogbg_molhiv/mapping/mol.csv.gz'):
        self.matching = pd.read_csv(filename)
        self.idx = 0

    def __call__(self, graph):
        mol = Chem.MolFromSmiles(self.matching.smiles[self.idx])
        self.idx += 1
        graph.mol = mol
        return graph


class OGB_Graph_Add_Mol:
    """
    Add rdkit mol object to OGB graph (hiv or pcba)
    """

    def __init__(self):
        pass

    def __call__(self, graph):
        # create a read/write molecule
        mol = Chem.RWMol()

        # set atom properties and add to the molecule
        for atom_ndx, feature in enumerate(graph.x):
            feature = feature.numpy().tolist()
            atom = Chem.Atom(feature[0] + 1)
            atom.SetChiralTag(get_chiral_tag(feature[1]))
            atom.SetFormalCharge(get_formal_charge(feature[3]))
            atom.SetNumRadicalElectrons(feature[5])
            atom.SetHybridization(get_hybridization(feature[6]))
            atom.SetIsAromatic(get_aromatic(feature[7]))
            mol.AddAtom(atom)

        bond_types = graph.edge_attr[:, 0].numpy()
        bonds = list(map(bond_ndx_to_bond, bond_types))

        # create a bond, set its properties and add to the molecule
        for ndx, (i, j) in enumerate(graph.edge_index.T):
            feature = graph.edge_attr[ndx].numpy().tolist()
            bond = mol.GetBondBetweenAtoms(i.item(), j.item())
            if bond is None:
                mol.AddBond(i.item(), j.item(), bonds[ndx])
                bond = mol.GetBondBetweenAtoms(i.item(), j.item())
                bond.SetStereo(get_bond_stereo(feature[1]))
                bond.SetIsConjugated(get_conjugated(feature[2]))

        # cleanup the molecule
        mol.UpdatePropertyCache()
        Chem.SanitizeMol(mol)

        # FastFindRings(mol)
        graph.mol = mol

        return graph
