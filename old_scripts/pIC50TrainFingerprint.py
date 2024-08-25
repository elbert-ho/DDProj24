import yaml
from tqdm import tqdm
import torch
from DiffusionModel import DiffusionModel
from pIC50Loader import pIC50Dataset
from pIC50Predictor import pIC50Predictor
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn

def train(model, train_dataloader, val_dataloader, criterion, optimizer, device, max_seq_length, d_model, num_epochs=100, patience=10):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0

        for noised_molecule, protein, time_step, pIC50 in train_dataloader:
            # Move tensors to device
            noised_molecule = noised_molecule.to(device)
            protein = protein.to(device).squeeze(1)
            time_step = time_step.to(device).squeeze(1)
            pIC50 = pIC50.to(device)

            noised_molecule = noised_molecule.reshape(-1, max_seq_length, d_model)
            
            mean_pool = noised_molecule.mean(dim=1)  # Mean pooling
            max_pool = noised_molecule.max(dim=1)[0]  # Max pooling
            first_token_last_layer = noised_molecule[:, 0, :]  # First token output of the last layer

            # Concatenate the vectors
            token_representations = torch.cat([mean_pool, max_pool, first_token_last_layer], dim=1)

            optimizer.zero_grad()
            outputs = model(token_representations, protein, time_step)
            loss = criterion(outputs.squeeze(), pIC50)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_dataloader)

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noised_molecule, protein, time_step, pIC50 in val_dataloader:
                # Move tensors to device
                noised_molecule = noised_molecule.to(device)
                protein = protein.to(device).squeeze(1)
                time_step = time_step.to(device).squeeze(1)
                pIC50 = pIC50.to(device)

                noised_molecule = noised_molecule.reshape(-1, max_seq_length, d_model)
            
                mean_pool = noised_molecule.mean(dim=1)  # Mean pooling
                max_pool = noised_molecule.max(dim=1)[0]  # Max pooling
                first_token_last_layer = noised_molecule[:, 0, :]  # First token output of the last layer
                # Concatenate the vectors
                token_representations = torch.cat([mean_pool, max_pool, first_token_last_layer], dim=1)

                outputs = model(token_representations, protein, time_step)
                loss = criterion(outputs.squeeze(), pIC50)
                val_loss += loss.item()

        val_loss /= len(val_dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print('Early stopping')
                break

# Load your dataset
protein_file = 'data/protein_embeddings.npy'
smiles_file = 'data/smiles_output.npy'
pIC50_file = 'data/pIC50.npy'

with open("hyperparams.yaml", "r") as file:
    config = yaml.safe_load(file)

num_diffusion_steps = config["diffusion_model"]["num_diffusion_steps"]

diffusion_model = DiffusionModel(unet_model=None, num_diffusion_steps=num_diffusion_steps)
dataset = pIC50Dataset(protein_file, smiles_file, pIC50_file, diffusion_model, num_diffusion_steps=num_diffusion_steps)

# Train-test split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders
batch_size = config["pIC50_model"]["batch_size"]
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model parameters
d_model = config["mol_model"]["d_model"]
max_seq_length = config["mol_model"]["max_seq_length"]

molecule_dim = d_model * 3
protein_dim = config["protein_model"]["protein_embedding_dim"]
hidden_dim = config["pIC50_model"]["hidden_dim"]
num_heads = config["pIC50_model"]["num_heads"]
lr = config["pIC50_model"]["lr"]
num_epochs = config["pIC50_model"]["num_epochs"]
patience = config["pIC50_model"]["patience"]
time_embed_dim = config["pIC50_model"]["time_embed_dim"]

# Instantiate model, criterion, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = pIC50Predictor(molecule_dim, protein_dim, hidden_dim, num_heads, time_embed_dim, num_diffusion_steps).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Train the model
train(model, train_dataloader, val_dataloader, criterion, optimizer, device, max_seq_length, d_model, num_epochs, patience)
torch.save(model.state_dict(), 'models/pIC50_model.pt')
