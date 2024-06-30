import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class pIC50Dataset(Dataset):
    def __init__(self, csv_file, diffusion_model, num_diffusion_steps):
        self.data = pd.read_csv(csv_file)
        self.diffusion_model = diffusion_model
        self.num_diffusion_steps = num_diffusion_steps

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        molecule = torch.tensor(eval(self.data.iloc[idx, 0]), dtype=torch.float32)
        protein = torch.tensor(eval(self.data.iloc[idx, 1]), dtype=torch.float32)
        pIC50 = torch.tensor(self.data.iloc[idx, 2], dtype=torch.float32)
        
        # Random time step
        time_step = torch.randint(0, self.num_diffusion_steps, (1,))
        
        # Noise the molecule
        noised_molecule = self.diffusion_model.noise_molecule(molecule, time_step)
        
        return noised_molecule, protein, time_step, pIC50