import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class pIC50Dataset(Dataset):
    def __init__(self, protein_file, smiles_file, pIC50_file, diffusion_model, num_diffusion_steps):
        self.protein_data = np.load(protein_file)
        self.smiles_data = np.load(smiles_file)
        self.pIC50_data = np.load(pIC50_file)
        self.diffusion_model = diffusion_model
        self.num_diffusion_steps = num_diffusion_steps

    def __len__(self):
        return len(self.pIC50_data)

    def __getitem__(self, idx):
        protein_cls = torch.tensor(self.protein_data[idx], dtype=torch.float32)
        smiles_string = torch.tensor(self.smiles_data[idx], dtype=torch.float32)
        pIC50 = torch.tensor(self.pIC50_data[idx], dtype=torch.float32)

        # Random time step
        time_step = torch.randint(0, self.num_diffusion_steps, (1,))
        # Noise the molecule
        noised_molecule = self.diffusion_model.noise_molecule(smiles_string, time_step)

        return noised_molecule, protein_cls, time_step, pIC50