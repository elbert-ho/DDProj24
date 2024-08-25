import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from MolPropLoader import MolPropDataset
from MolPropModel import PropertyModel
from DiffusionModelGLIDE import *
import torch.nn as nn
from tqdm import tqdm
import yaml

# Load the dataset

with open("hyperparams.yaml", "r") as file:
    config = yaml.safe_load(file)

num_diffusion_steps = config["diffusion_model"]["num_diffusion_steps"]

diffusion_model = GaussianDiffusion(betas=get_named_beta_schedule(num_diffusion_steps))
dataset = MolPropDataset(smiles_file='data/smiles_output_selfies_normal.npy', qed_file='data/qed.npy', sas_file='data/sas.npy')
# Split into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
batch_size = config["prop_model"]["batch_size"]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize models
input_size = config["mol_model"]["d_model"] * config["mol_model"]["max_seq_length"]
lr = config["prop_model"]["learning_rate"]
time_embed_dim = config["prop_model"]["time_embed_dim"]
d_model = config["mol_model"]["d_model"]
max_seq_length = config["mol_model"]["max_seq_length"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


qed_model = PropertyModel(input_size, num_diffusion_steps, time_embed_dim).to(device)
sas_model = PropertyModel(input_size, num_diffusion_steps, time_embed_dim).to(device)

# Define loss function and optimizers
criterion = nn.MSELoss()
qed_optimizer = optim.Adam(qed_model.parameters(), lr=lr)
sas_optimizer = optim.Adam(sas_model.parameters(), lr=lr)

# Early stopping parameters
patience = config["prop_model"]["patience"]
best_val_loss = float('inf')
best_val_qed = float('inf')
best_val_sas = float('inf')
patience_qed = 0
patience_sas = 0

# Training and validation loop
epoch = 0
while patience_qed < patience or patience_sas < patience:
    epoch += 1
    # Training phase
    qed_model.train()
    sas_model.train()
    train_qed_loss = 0.0
    train_sas_loss = 0.0

    for inputs, qeds, sas in tqdm(train_loader, desc=f"Epoch {epoch} [Training]"):
        # Move tensors to device
        inputs, qeds, sas = inputs.to(device), qeds.to(device), sas.to(device)
        b = inputs.shape[0]
        time_step = torch.randint(0, 1000, [b,], device=device)
        
        inputs = diffusion_model.q_sample(inputs, time_step)

        # Zero the parameter gradients

        if patience_qed < patience:
            qed_optimizer.zero_grad()
            qed_outputs = qed_model(inputs, time_step).clamp(0, 1)
            qed_loss = criterion(qed_outputs.squeeze(), qeds)
            qed_loss.backward()
            qed_optimizer.step()
            train_qed_loss += qed_loss.item() * inputs.size(0)

        if patience_sas < patience:
            sas_optimizer.zero_grad()
            # Forward pass
            sas_outputs = sas_model(inputs, time_step).clamp(1, 10)
            # Calculate loss
            sas_loss = criterion(sas_outputs.squeeze(), sas)
            # Backward pass and optimize
            sas_loss.backward()
            sas_optimizer.step()
            train_sas_loss += sas_loss.item() * inputs.size(0)
    
    train_qed_loss /= len(train_loader.dataset)
    train_sas_loss /= len(train_loader.dataset)

    # Validation phase
    qed_model.eval()
    sas_model.eval()
    val_qed_loss = 0.0
    val_sas_loss = 0.0

    with torch.no_grad():
        for inputs, qeds, sas in tqdm(val_loader, desc=f"Epoch {epoch} [Validation]"):
            # Move tensors to device
            inputs, qeds, sas = inputs.to(device), qeds.to(device), sas.to(device)
            b = inputs.shape[0]
            time_step = torch.randint(0, 1000, [b,], device=device)
            inputs = diffusion_model.q_sample(inputs, time_step)

            if patience_qed < patience:
                qed_outputs = qed_model(inputs, time_step)
                qed_loss = criterion(qed_outputs.squeeze(), qeds)
                val_qed_loss += qed_loss.item() * inputs.size(0)

            if patience_sas < patience:
                sas_outputs = sas_model(inputs, time_step)
                sas_loss = criterion(sas_outputs.squeeze(), sas)
                val_sas_loss += sas_loss.item() * inputs.size(0)

    val_qed_loss /= len(val_loader.dataset)
    val_sas_loss /= len(val_loader.dataset)

    print(f'Epoch [{epoch}], '
          f'Train QED Loss: {train_qed_loss:.4f}, Val QED Loss: {val_qed_loss:.4f}, '
          f'Train SAS Loss: {train_sas_loss:.4f}, Val SAS Loss: {val_sas_loss:.4f}')

    # Check early stopping condition
    # avg_val_loss = (val_qed_loss + val_sas_loss) / 2
    # if avg_val_loss < best_val_loss:
    #     best_val_loss = avg_val_loss
    #     patience_counter = 0
    # else:
    #     patience_counter += 1
    
    if val_qed_loss < best_val_qed:
        best_val_qed = val_qed_loss
        torch.save(qed_model.state_dict(), 'models/qed_model.pt')

    if val_sas_loss < best_val_sas:
        best_val_sas = val_sas_loss
        torch.save(sas_model.state_dict(), 'models/sas_model.pt')
