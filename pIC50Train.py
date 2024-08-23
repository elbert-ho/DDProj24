import yaml
from tqdm import tqdm
import torch
from pIC50Loader import pIC50Dataset
from pIC50Predictor2 import pIC50Predictor
from DiffusionModelGLIDE import *
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn

def train(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs=100, patience=10):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    progress_bar = tqdm(range(num_epochs))

    for epoch in progress_bar:
        model.train()
        train_loss = 0.0

        for mol, protein, pIC50 in train_dataloader:
            b = mol.shape[0]
            mol = mol.to(device)
            protein = protein.to(device).squeeze(1)
            time_step = torch.randint(0, 1000, [b,], device=device)
            mol = diffusion_model.q_sample(mol, time_step)
            pIC50 = pIC50.to(device)

            optimizer.zero_grad()
            outputs = model(mol, protein, time_step)
            loss = criterion(outputs.squeeze(), pIC50)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for mol, protein, pIC50 in val_dataloader:
                b = mol.shape[0]
                mol = mol.to(device)
                protein = protein.to(device).squeeze(1)
                time_step = torch.randint(0, 1000, [b,], device=device)
                mol = diffusion_model.q_sample(mol, time_step)
                pIC50 = pIC50.to(device)
                outputs = model(mol, protein, time_step)
                loss = criterion(outputs.squeeze(), pIC50)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)

        train_loss = math.sqrt(train_loss)
        val_loss = math.sqrt(val_loss)

        # print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        progress_bar.set_description(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'models/pIC50_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                # print('Early stopping')
                break

# Load your dataset
# Ensure you have defined the DiffusionModel and pIC50Dataset classes
protein_file = 'data/protein_embeddings.npy'
smiles_file = 'data/smiles_output_selfies_normal.npy'
pIC50_file = 'data/pIC50.npy'

with open("hyperparams.yaml", "r") as file:
    config = yaml.safe_load(file)

num_diffusion_steps = config["diffusion_model"]["num_diffusion_steps"]
diffusion_model = GaussianDiffusion(betas=get_named_beta_schedule(num_diffusion_steps))
dataset = pIC50Dataset(protein_file, smiles_file, pIC50_file)

# Train-test split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
generator1 = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator1)

# DataLoaders
batch_size = config["pIC50_model"]["batch_size"]
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model parameters
molecule_dim = config["mol_model"]["d_model"] * config["mol_model"]["max_seq_length"]
protein_dim = config["protein_model"]["protein_embedding_dim"]
hidden_dim = config["pIC50_model"]["hidden_dim"]
num_heads = config["pIC50_model"]["num_heads"]
lr = config["pIC50_model"]["lr"]
num_epochs = config["pIC50_model"]["num_epochs"]
patience = config["pIC50_model"]["patience"]
time_embed_dim = config["pIC50_model"]["time_embed_dim"]

# Instantiate model, criterion, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = pIC50Predictor(molecule_dim, protein_dim, num_heads, time_embed_dim, num_diffusion_steps).to(device)
model = pIC50Predictor(num_diffusion_steps, protein_dim).to(device)
# model.load_state_dict(torch.load('models/pIC50_model.pt', map_location=device))

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Train the model
train(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs, patience)
# torch.save(model.state_dict(), 'models/pIC50_model.pt')
