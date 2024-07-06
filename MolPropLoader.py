import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class MolPropDataset(Dataset):
    def __init__(self, smiles_file, qed_file, sas_file, diffusion_model, num_diffusion_steps):
        self.smiles = np.load(smiles_file)
        self.qeds = np.load(qed_file)
        self.sas = np.load(sas_file)
        self.num_diffusion_steps = num_diffusion_steps
        self.diffusion_model = diffusion_model

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles_enc = torch.tensor(self.smiles[idx], dtype=torch.float32).flatten()
        qed = torch.tensor(self.qeds[idx], dtype=torch.float32)
        sas = torch.tensor(self.sas[idx], dtype=torch.float32)
        # Random time step
        time_step = torch.randint(0, self.num_diffusion_steps, (1,))
        # Noise the molecule
        noised_molecule = self.diffusion_model.noise_molecule(smiles_enc, time_step)

        return noised_molecule, time_step, qed, sas