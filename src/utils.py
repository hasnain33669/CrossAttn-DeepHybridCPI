import random
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from torch_geometric.data import Batch

# Protein vocabulary
VOCAB_PROTEIN = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
    "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
    "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25
}


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomPiecewiseActivation(nn.Module):
    """Custom activation function: x if x >= 0 else sin(x)."""
    
    def forward(self, x):
        return torch.where(x >= 0, x, torch.sin(x))


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    """Collate function for DataLoader."""
    return Batch.from_data_list(batch)


def one_of_k_encoding(x, allowable_set):
    """One-hot encoding with allowed set."""
    if x not in allowable_set:
        raise Exception(f"input {x} not in allowable set{allowable_set}")
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """One-hot encoding with unknown handling."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    """Generate features for a given atom."""
    encoding = one_of_k_encoding_unk(atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
         'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
         'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
         'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
         'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'])

    encoding += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    encoding += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    encoding += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    encoding += one_of_k_encoding_unk(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        'other'
    ])
    encoding += [atom.GetIsAromatic()]

    try:
        encoding += one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R', 'S'])
        encoding += [atom.HasProp('_ChiralityPossible')]
    except:
        encoding += [0, 0] + [atom.HasProp('_ChiralityPossible')]

    return np.array(encoding, dtype=np.float32)


def mol_to_graph(mol):
    """Convert RDKit molecule to graph representation."""
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        feature = feature / (np.sum(feature) + 1e-8)
        features.append(feature)

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    if len(edges) == 0:
        return features, [[0, 0]]

    import networkx as nx
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return features, edge_index
