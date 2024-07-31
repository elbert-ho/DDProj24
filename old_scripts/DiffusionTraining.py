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
from tqdm import tqdm


def train_model(dataset, epochs=10, batch_size=32, lr=1e-4, lambda_vlb=0.001, num_diffusion_steps=1000, diffusion_model=None):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(diffusion_model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(epochs):
        print(f"EPOCH {epoch + 1} BEGIN")
        epoch_loss = 0
        batch_count = 0
        
        for original_molecule, protein_embedding, time_step, _ in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
            original_molecule = original_molecule.to(device)
            protein_embedding = protein_embedding.to(device)
            time_step = time_step.to(device)
            
            optimizer.zero_grad()
            
            noise = torch.randn_like(original_molecule)
            noised_molecule = diffusion_model.noise_molecule(original_molecule, time_step, noise=noise)
            
            log_var, noise_pred = diffusion_model(noised_molecule, time_step, protein_embedding)
            
            # Compute L_simple
            L_simple = torch.mean((noise_pred - noise) ** 2)

            # Compute q_mean and L_vlb for each timestep
            alpha_bar_t = diffusion_model.alpha_bar[time_step].unsqueeze(-1).unsqueeze(-1)  # Adjust dimensions for broadcasting
            q_mean = (noised_molecule - torch.sqrt(1 - alpha_bar_t) * noise) / torch.sqrt(alpha_bar_t)
            L_vlb_t = 0.5 * torch.sum(log_var + (noised_molecule - q_mean) ** 2 / torch.exp(log_var) - 1, dim=1).mean()

            # Add contributions for each timestep to L_vlb
            L_vlb = L_vlb_t

            # Compute L_0
            L_0 = -diffusion_model.log_prob(original_molecule, noise_pred)

            # Compute L_T
            prior_mean = torch.zeros_like(original_molecule)
            prior_log_var = torch.zeros_like(original_molecule)
            L_T = 0.5 * torch.sum(prior_log_var + (original_molecule - prior_mean) ** 2 / torch.exp(prior_log_var) - 1, dim=1).mean()

            # Total VLB Loss
            L_vlb_total = L_0 + L_vlb + L_T

            # Compute L_hybrid
            L_hybrid = L_simple + lambda_vlb * L_vlb_total

            L_hybrid.backward()
            optimizer.step()

            epoch_loss += L_hybrid.item()
            batch_count += 1

        average_loss = epoch_loss / batch_count
        print(f"Epoch [{epoch + 1}/{epochs}], Average Loss: {average_loss:.4f}")

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
device = config["mol_model"]["device"]

# Initialize models
unet_model = UNet1D(input_channels=1, output_channels=1, time_embedding_dim=unet_ted, protein_embedding_dim=protein_embedding_dim, num_diffusion_steps=num_diffusion_steps, device=device).to(device)

# Assuming original_molecule data is available in the dataset
diffusion_model = DiffusionModel(unet_model=unet_model, num_diffusion_steps=num_diffusion_steps, device=device)
dataset = pIC50Dataset('data/protein_embeddings.npy', 'data/smiles_output.npy', 'data/pic50.npy', diffusion_model=diffusion_model, num_diffusion_steps=num_diffusion_steps)
diffusion_model = diffusion_model.to(device)

# Train the model
print("BEGIN TRAIN")
trained_diffusion_model = train_model(dataset, epochs=epochs, batch_size=batch_size, lr=lr, lambda_vlb=lambda_vlb, num_diffusion_steps=num_diffusion_steps, diffusion_model=diffusion_model)
exit()
torch.save(unet_model.state_dict(), 'models/unet_model.pt')
