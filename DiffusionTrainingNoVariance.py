import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tokenizers import Tokenizer
from DiffusionModel import DiffusionModel
from pIC50Loader import pIC50Dataset
from UNet import UNet1D
import yaml
from tqdm import tqdm
import numpy as np

def train_model(dataset, epochs=100, batch_size=32, lr=1e-4, num_diffusion_steps=1000, diffusion_model=None, patience=10):
    # Split dataset into training and validation sets (80-20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.Adam(diffusion_model.parameters(), lr=lr)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        print(f"EPOCH {epoch + 1} BEGIN")
        diffusion_model.train()
        epoch_loss = 0
        batch_count = 0

        for original_molecule, protein_embedding, time_step, _ in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
            original_molecule = original_molecule.to(device)
            protein_embedding = protein_embedding.to(device)
            time_step = time_step.to(device)
            
            optimizer.zero_grad()

            noise = torch.randn_like(original_molecule)
            noised_molecule = diffusion_model.noise_molecule(original_molecule, time_step, noise=noise)
            
            _, noise_pred = diffusion_model(noised_molecule, time_step, protein_embedding)
            
            # Compute L_simple
            L_simple = torch.mean((noise_pred - noise) ** 2)

            L_simple.backward()
            optimizer.step()

            epoch_loss += L_simple.item()
            batch_count += 1

        average_train_loss = epoch_loss / batch_count
        print(f"Epoch [{epoch + 1}/{epochs}], Average Training Loss: {average_train_loss:.4f}")

        # Validation
        diffusion_model.eval()
        val_loss = 0
        with torch.no_grad():
            for original_molecule, protein_embedding, time_step, _ in tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}/{epochs}', leave=False):
                original_molecule = original_molecule.to(device)
                protein_embedding = protein_embedding.to(device)
                time_step = time_step.to(device)
                
                noise = torch.randn_like(original_molecule)
                noised_molecule = diffusion_model.noise_molecule(original_molecule, time_step, noise=noise)
                
                _, noise_pred = diffusion_model(noised_molecule, time_step, protein_embedding)
                
                L_simple = torch.mean((noise_pred - noise) ** 2)
                val_loss += L_simple.item()
        
        average_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Average Validation Loss: {average_val_loss:.4f}")

        # Early stopping
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            patience_counter = 0
            torch.save(diffusion_model.state_dict(), 'best_diffusion_model.pt')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    return diffusion_model

with open("hyperparams.yaml", "r") as file:
    config = yaml.safe_load(file)

protein_embedding_dim = config["protein_model"]["protein_embedding_dim"]
num_diffusion_steps = config["diffusion_model"]["num_diffusion_steps"]
batch_size = config["diffusion_model"]["batch_size"]
lr = config["diffusion_model"]["lr"]
epochs = config["diffusion_model"]["epochs"]
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
trained_diffusion_model = train_model(dataset, epochs=epochs, batch_size=batch_size, lr=lr, num_diffusion_steps=num_diffusion_steps, diffusion_model=diffusion_model)
torch.save(unet_model.state_dict(), 'models/unet_model_no_var.pt')
