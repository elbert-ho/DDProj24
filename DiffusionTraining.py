import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tokenizers import Tokenizer
from DiffusionModel import DiffusionModel
from pIC50Loader import pIC50Dataset
from UNet import UNet1D
from MolTransformer import MultiTaskTransformer
from pIC50Predictor import pIC50Predictor
from MolPropModel import PropertyModel
import yaml

def train_model(unet_model, dataset, epochs=10, batch_size=32, lr=1e-4, lambda_vlb=.001, num_diffusion_steps=1000):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    diffusion_model = DiffusionModel(unet_model, num_diffusion_steps)
    
    optimizer = optim.Adam(diffusion_model.parameters(), lr=lr)

    for epoch in range(epochs):
        for original_molecule, protein_embedding, time_step, _ in dataloader:
            optimizer.zero_grad()
            
            noise = torch.randn_like(original_molecule)
            noised_molecule = diffusion_model.noise_molecule(original_molecule, time_step, noise=noise)
            
            log_var, noise_pred = diffusion_model(noised_molecule, time_step, protein_embedding)
            
            # Compute L_simple
            L_simple = torch.mean((noise_pred - noise) ** 2)
            
            # Compute L_vlb
            q_mean = (noised_molecule - torch.sqrt(1 - diffusion_model.alpha_bar[time_step]) * noise) / torch.sqrt(diffusion_model.alpha_bar[time_step])
            L_vlb = 0.5 * torch.sum(log_var + (noised_molecule - q_mean) ** 2 / torch.exp(log_var) - 1, dim=1).mean()
            
            # Compute L_hybrid
            L_hybrid = L_simple + lambda_vlb * L_vlb
            
            L_hybrid.backward()
            optimizer.step()

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {L_hybrid.item()}")

    return diffusion_model

with open("hyperparams.yaml", "r") as file:
    config = yaml.safe_load(file)

protein_embedding_dim = config["protein_model"]["protein_embedding_dim"]
num_diffusion_steps = config["diffusion_model"]["num_diffusion_steps"]
batch_size = config["diffusion_model"]["batch_size"]
lr = config["diffusion_model"]["lr"]
epochs = config["diffusion_model"]["epochs"]
patience = config["diffusion_model"]["patience"]
lambda_vlb = config["diffusion_model"]["lambda_vlb"]
unet_ted = config["UNet"]["time_embedding_dim"]

# Initialize models
unet_model = UNet1D(input_channels=1, output_channels=1, time_embedding_dim=unet_ted, protein_embedding_dim=protein_embedding_dim, num_diffusion_steps=num_diffusion_steps)

# Assuming original_molecule data is available in the dataset
dataset = pIC50Dataset('protein_data.npy', 'smiles_data.npy', 'pic50_data.npy', num_diffusion_steps=num_diffusion_steps)

# Train the model
trained_diffusion_model = train_model(unet_model, dataset, epochs=epochs, batch_size=batch_size, lr=lr, lambda_vlb=lambda_vlb, num_diffusion_steps=num_diffusion_steps)
