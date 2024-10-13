import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

class ProtLigDataset(Dataset):
    def __init__(self, id_file, att_file, smiles_file, id_file2=None, att_file2=None, smiles_file2=None):
        self.ids = np.load(id_file)
        self.atts = np.load(att_file)

        self.smiles_data = np.load(smiles_file)
        if id_file2:
            # print(self.protein_data.shape)
            # self.protein_data = np.append(self.protein_data, pd.read_csv(protein_file2)["Protein Sequence"].to_numpy(), axis=0)
            # print(self.protein_data.shape)
            # exit()
            self.ids = np.append(self.ids, np.load(id_file2), axis=0)

        if smiles_file2:
            # print(self.smiles_data.shape)
            self.smiles_data = np.append(self.smiles_data, np.load(smiles_file2), axis=0)
            # print(self.smiles_data.shape)
            # exit()

        if att_file2:
            self.atts = np.append(self.atts, np.load(att_file2), axis=0)


        self.ids = torch.tensor(self.ids, dtype=torch.int32)
        self.atts = torch.tensor(self.atts, dtype=torch.int32)

        # JUST CHANGE THIS LINE
        self.smiles_data = torch.tensor(self.smiles_data, dtype=torch.float32).reshape(-1, 256, 128)


    def __len__(self):
        return len(self.smiles_data)

    def __getitem__(self, idx):
        ids = self.ids[idx]
        smiles = self.smiles_data[idx]
        atts = self.atts[idx]
        # print(smiles.shape)
        return smiles, ids, atts