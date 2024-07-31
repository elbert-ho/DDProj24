import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ProtLigDataset(Dataset):
    def __init__(self, protein_file, smiles_file):
        self.protein_data = np.load(protein_file)
        self.smiles_data = np.load(smiles_file)

    def __len__(self):
        return len(self.smiles_data)

    def __getitem__(self, idx):
        protein_cls = torch.tensor(self.protein_data[idx], dtype=torch.float32)
        smiles = torch.tensor(self.smiles_data[idx], dtype=torch.float32)
        
        return smiles, protein_cls