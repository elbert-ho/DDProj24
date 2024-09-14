import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ProtLigDataset(Dataset):
    def __init__(self, protein_file, smiles_file, protein_file2=None, smiles_file2=None):
        self.protein_data = np.load(protein_file)
        self.smiles_data = np.load(smiles_file)
        if protein_file2:
            # print(self.protein_data.shape)
            self.protein_data = np.append(self.protein_data, np.load(protein_file2), axis=0)
            # print(self.protein_data.shape)
            # exit()

        if smiles_file2:
            # print(self.smiles_data.shape)
            self.smiles_data = np.append(self.smiles_data, np.load(smiles_file2), axis=0)
            # print(self.smiles_data.shape)
            # exit()

    def __len__(self):
        return len(self.smiles_data)

    def __getitem__(self, idx):
        protein_cls = torch.tensor(self.protein_data[idx], dtype=torch.float32)
        smiles = torch.tensor(self.smiles_data[idx], dtype=torch.float32).reshape(256, 128)
        # print(smiles.shape)
        return smiles, protein_cls