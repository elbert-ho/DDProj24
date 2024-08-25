import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class pIC50Dataset(Dataset):
    def __init__(self, protein_file, smiles_file, pIC50_file):
        self.protein_data = np.load(protein_file)
        self.smiles_data = np.load(smiles_file)
        self.pIC50_data = np.load(pIC50_file)

    def __len__(self):
        return len(self.pIC50_data)

    def __getitem__(self, idx):
        protein_cls = torch.tensor(self.protein_data[idx], dtype=torch.float32)
        smiles_string = torch.tensor(self.smiles_data[idx], dtype=torch.float32).reshape(256, 128)
        pIC50 = torch.tensor(self.pIC50_data[idx], dtype=torch.float32)
        return smiles_string, protein_cls, pIC50