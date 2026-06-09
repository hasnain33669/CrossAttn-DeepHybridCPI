
import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data, Batch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from rdkit import Chem
import networkx as nx
from tqdm import tqdm

from src.utils import atom_features, mol_to_graph, VOCAB_PROTEIN


def load_cpi_data(data_path='data.txt'):
    """Load CPI data from text file."""
    print("="*60)
    print("LOADING CPI DATA")
    print("="*60)

    if not os.path.exists(data_path):
        if os.path.exists('/content/data.txt'):
            data_path = '/content/data.txt'
        else:
            print(f"❌ Error: {data_path} not found!")
            return None

    print(f"Loading data from: {data_path}")

    smiles_list, protein_list, label_list = [], [], []

    with open(data_path, 'r') as f:
        lines = f.readlines()

    print(f"Total lines in file: {len(lines)}")

    invalid_count = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) >= 3:
            smiles, protein, label = parts[0], parts[1], parts[2]
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    smiles_list.append(smiles)
                    protein_list.append(protein)
                    label_list.append(int(label))
                else:
                    invalid_count += 1
            except:
                invalid_count += 1
        else:
            invalid_count += 1

    df = pd.DataFrame({
        'compound_iso_smiles': smiles_list,
        'target_sequence': protein_list,
        'affinity': label_list
    })

    print(f"\n✅ Loaded {len(df)} valid samples")
    print(f"❌ Skipped {invalid_count}")
    print(f"Positive: {df['affinity'].sum()} | Negative: {len(df) - df['affinity'].sum()}")

    return df


def split_data(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """Split data into train, validation, and test sets with stratification."""
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        stratify=df['affinity']
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        random_state=random_state,
        stratify=temp_df['affinity']
    )

    print(f"\n📊 Data Split Summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")

    print(f"\n  Train - Positive: {train_df['affinity'].sum()}, Negative: {len(train_df) - train_df['affinity'].sum()}")
    print(f"  Val   - Positive: {val_df['affinity'].sum()}, Negative: {len(val_df) - val_df['affinity'].sum()}")
    print(f"  Test  - Positive: {test_df['affinity'].sum()}, Negative: {len(test_df) - test_df['affinity'].sum()}")

    return train_df, val_df, test_df


def generate_esm2_embeddings(sequences, model_name='esm2_t30_150M_UR50D', batch_size=1, device='cpu'):
    """Generate ESM-2 embeddings for protein sequences."""
    import esm
    import torch

    print(f"Loading ESM-2 model ({model_name})...")

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    batch_converter = alphabet.get_batch_converter()
    model.eval()

    device = torch.device(device)
    model = model.to(device)

    embeddings_dict = {}
    sequences_list = list(sequences)

    print(f"Generating embeddings for {len(sequences_list)} unique protein sequences...")

    for i in tqdm(range(0, len(sequences_list), batch_size), desc="ESM-2 Embeddings"):
        batch_seqs = sequences_list[i:i+batch_size]
        batch_data = [(f"protein_{j}", seq) for j, seq in enumerate(batch_seqs)]

        batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[30])
            token_representations = results["representations"][30]

            for j, seq in enumerate(batch_seqs):
                seq_len = len(seq)
                residue_embeddings = token_representations[j, 1:seq_len+1, :]
                seq_embedding = residue_embeddings.mean(dim=0)
                embeddings_dict[seq] = seq_embedding.cpu()

    print(f"✅ Generated embeddings for {len(embeddings_dict)} sequences")
    return embeddings_dict


class CrossAttnCPIDataset(InMemoryDataset):
    """Dataset class for CPI prediction."""
    
    def __init__(self, root, types='train', protein_embeddings=None, transform=None, pre_transform=None):
        self.types = types
        self.protein_embeddings = protein_embeddings
        super().__init__(root, transform, pre_transform)

        if types == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        elif types == 'val':
            self.data, self.slices = torch.load(self.processed_paths[1], weights_only=False)
        elif types == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2], weights_only=False)

    @property
    def raw_file_names(self):
        return ['data_train.csv', 'data_val.csv', 'data_test.csv']

    @property
    def processed_file_names(self):
        return ['processed_data_train.pt', 'processed_data_val.pt', 'processed_data_test.pt']

    def download(self):
        pass

    def process_data(self, data_path, graph_dict):
        df = pd.read_csv(data_path)
        data_list = []
        delete_list = []

        for i, row in df.iterrows():
            smi = row['compound_iso_smiles']
            sequence = row['target_sequence']
            label = row['affinity']

            if graph_dict.get(smi) is None:
                delete_list.append(i)
                continue

            x, edge_index = graph_dict[smi]

            if sequence in self.protein_embeddings:
                protein_embedding = self.protein_embeddings[sequence].unsqueeze(0)
            else:
                protein_embedding = torch.zeros(1, 640)

            data = Data(
                x=torch.FloatTensor(x),
                edge_index=torch.LongTensor(edge_index).t().contiguous(),
                y=torch.LongTensor([label]),
                protein_embedding=protein_embedding
            )
            data_list.append(data)

        if delete_list:
            df = df.drop(delete_list, axis=0, inplace=False)
            df.to_csv(data_path, index=False)

        return data_list

    def process(self):
        df_train = pd.read_csv(self.raw_paths[0])
        df_val = pd.read_csv(self.raw_paths[1])
        df_test = pd.read_csv(self.raw_paths[2])
        df = pd.concat([df_train, df_val, df_test])
        smiles = df['compound_iso_smiles'].unique()

        graph_dict = {}
        for smile in tqdm(smiles, desc="Building molecular graphs"):
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                continue
            graph_dict[smile] = mol_to_graph(mol)

        train_list = self.process_data(self.raw_paths[0], graph_dict)
        val_list = self.process_data(self.raw_paths[1], graph_dict)
        test_list = self.process_data(self.raw_paths[2], graph_dict)

        data, slices = self.collate(train_list)
        torch.save((data, slices), self.processed_paths[0])

        data, slices = self.collate(val_list)
        torch.save((data, slices), self.processed_paths[1])

        data, slices = self.collate(test_list)
        torch.save((data, slices), self.processed_paths[2])

        print(f"Processing complete! Train: {len(train_list)}, Val: {len(val_list)}, Test: {len(test_list)}")
