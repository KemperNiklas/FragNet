import os
import sys

import networkx as nx
from rdkit import Chem
from rdkit.Chem import Descriptors, RDConfig
from torch_geometric.transforms import BaseTransform

#import sascorer
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
from rdkit.Contrib.SA_Score import sascorer


def penalized_logp(mol):
    """
    Reward that consists of log p penalized by SA and # long cycles,
    as described in (Kusner et al. 2017). Scores are normalized based on the
    statistics of 250k_rndm_zinc_drugs_clean.smi dataset
    :param mol: rdkit mol object
    :return: float
    """
    # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0485696876403053
    cycle_std = 0.2860212110245455

    log_p = Descriptors.MolLogP(mol)
    SA = -sascorer.calculateScore(mol)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(
        Chem.rdmolops.GetAdjacencyMatrix(mol)))
    # print(cycle_list)
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std

    y = normalized_log_p + normalized_SA + normalized_cycle

    return normalized_log_p, normalized_SA, normalized_cycle, y


def num_cycles(mol):
    rings = Chem.GetSymmSSSR(mol)
    num_cycles = 0
    for ring in rings:
        if len(ring) > 6:
            num_cycles += 1
    return num_cycles


def penalized_logp_fix(mol):
    """
    Reward that consists of log p penalized by SA and # long cycles,
    as described in (Kusner et al. 2017). Scores are normalized based on the
    statistics of 250k_rndm_zinc_drugs_clean.smi dataset.
    This version fixes the cycle score to be as described in the paper (*number* of cycles > 6)
    :param mol: rdkit mol object
    :return: float
    """
    # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095
    cycle_mean = -0.0347
    cycle_std = 0.03509591

    log_p = Descriptors.MolLogP(mol)
    SA = -sascorer.calculateScore(mol)

    cycle_score = -num_cycles(mol)

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    normalized_cycle = (cycle_score - cycle_mean) / cycle_std

    y = normalized_log_p + normalized_SA + normalized_cycle

    return normalized_log_p, normalized_SA, normalized_cycle, y


def logp(mol):
    # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
    logP_mean = 2.4570953396190123
    logP_std = 1.434324401111988
    SA_mean = -3.0525811293166134
    SA_std = 0.8335207024513095

    log_p = Descriptors.MolLogP(mol)
    SA = -sascorer.calculateScore(mol)

    normalized_log_p = (log_p - logP_mean) / logP_std
    normalized_SA = (SA - SA_mean) / SA_std
    # normalized_cycle = (cycle_score - cycle_mean) / cycle_std

    y = normalized_log_p + normalized_SA

    return normalized_log_p, normalized_SA, y


class FixPlogP(BaseTransform):
    def __init__(self):
        super(FixPlogP, self).__init__()

    def __call__(self, data):
        data.y = penalized_logp_fix(data.mol)[3]
        return data


class LogP(BaseTransform):
    def __init__(self):
        super(LogP, self).__init__()

    def __call__(self, data):
        data.y = logp(data.mol)[2]
        return data
