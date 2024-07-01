import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MolPropDataset(Dataset):
    def __init__(self, smiles_fingerprint_file, qed_file, sas_file, num_diffusion_steps):
        self.smiles_fingerprints = np.load(smiles_fingerprint_file)
        self.qeds = np.load(qed_file)
        self.sas = np.load(sas_file)

    def __len__(self):
        return len(self.smiles_fingerprints)

    def __getitem__(self, idx):
        smiles_fingerprint = torch.tensor(self.smiles_fingerprints[idx], dtype=torch.float32)
        qed = torch.tensor(self.qeds[idx], dtype=torch.float32)
        sas = torch.tensor(self.sas[idx], dtype=torch.float32)
        return smiles_fingerprint, qed, sas