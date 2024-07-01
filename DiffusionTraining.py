import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from DiffusionModel import DiffusionModel
from pIC50Loader import pIC50Dataset
from UNet import UNet1D

def train_model(unet_model, dataset, epochs=10, batch_size=32, lr=1e-4, lambda_vlb=1.0):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    diffusion_model = DiffusionModel(unet_model)
    
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


# Initialize models
unet_model = UNet1D(input_channels=1, output_channels=1, time_embedding_dim=128, protein_embedding_dim=512)

# Assuming original_molecule data is available in the dataset
dataset = pIC50Dataset('protein_data.npy', 'smiles_data.npy', 'pic50_data.npy', num_diffusion_steps=1000)

# Train the model
trained_diffusion_model = train_model(unet_model, dataset)
