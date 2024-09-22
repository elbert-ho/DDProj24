import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from MolPropLoader import MolPropDataset
from MolPropModel import PropertyModel
from DiffusionModel import DiffusionModel
import torch.nn as nn
from tqdm import tqdm
import yaml

# Load the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the dataset and configuration
with open("hyperparams.yaml", "r") as file:
    config = yaml.safe_load(file)

num_diffusion_steps = config["diffusion_model"]["num_diffusion_steps"]

diffusion_model = DiffusionModel(unet_model=None, num_diffusion_steps=num_diffusion_steps)

dataset = MolPropDataset(smiles_file='data/smiles_output.npy', qed_file='data/qed.npy', sas_file='data/sas.npy', diffusion_model=diffusion_model, num_diffusion_steps=num_diffusion_steps)

# Split into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
batch_size = config["prop_model"]["batch_size"]
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize models
input_size = config["mol_model"]["d_model"] * 3
lr = config["prop_model"]["learning_rate"]
time_embed_dim = config["prop_model"]["time_embed_dim"]
d_model = config["mol_model"]["d_model"]
max_seq_length = config["mol_model"]["max_seq_length"]

qed_model = PropertyModel(input_size, num_diffusion_steps, time_embed_dim).to(device)
sas_model = PropertyModel(input_size, num_diffusion_steps, time_embed_dim).to(device)

# Define loss function and optimizers
criterion = nn.MSELoss()
qed_optimizer = optim.Adam(qed_model.parameters(), lr=lr)
sas_optimizer = optim.Adam(sas_model.parameters(), lr=lr)

# Early stopping parameters
patience = config["prop_model"]["patience"]
best_val_loss = float('inf')
patience_counter = 0

# Training and validation loop
epoch = 0
while patience_counter < patience:
    epoch += 1
    # Training phase
    qed_model.train()
    sas_model.train()
    train_qed_loss = 0.0
    train_sas_loss = 0.0

    for inputs, time_step, qeds, sas in tqdm(train_loader, desc=f"Epoch {epoch} [Training]"):
        # Move tensors to device
        inputs, time_step, qeds, sas = inputs.to(device), time_step.to(device), qeds.to(device), sas.to(device)

        # Zero the parameter gradients
        qed_optimizer.zero_grad()
        sas_optimizer.zero_grad()

        noised_molecule = inputs.reshape(-1, max_seq_length, d_model)
        
        mean_pool = noised_molecule.mean(dim=1)  # Mean pooling
        max_pool = noised_molecule.max(dim=1)[0]  # Max pooling
        first_token_last_layer = noised_molecule[:, 0, :]  # First token output of the last layer

        # Concatenate the vectors
        token_representations = torch.cat([mean_pool, max_pool, first_token_last_layer], dim=1)

        # Forward pass
        qed_outputs = qed_model(token_representations, time_step)
        sas_outputs = sas_model(token_representations, time_step)

        # Calculate loss
        qed_loss = criterion(qed_outputs.squeeze(), qeds)
        sas_loss = criterion(sas_outputs.squeeze(), sas)

        # Backward pass and optimize
        qed_loss.backward()
        qed_optimizer.step()

        sas_loss.backward()
        sas_optimizer.step()

        train_qed_loss += qed_loss.item() * inputs.size(0)
        train_sas_loss += sas_loss.item() * inputs.size(0)

    train_qed_loss /= len(train_loader.dataset)
    train_sas_loss /= len(train_loader.dataset)

    # Validation phase
    qed_model.eval()
    sas_model.eval()
    val_qed_loss = 0.0
    val_sas_loss = 0.0

    with torch.no_grad():
        for inputs, time_step, qeds, sas in tqdm(val_loader, desc=f"Epoch {epoch} [Validation]"):
            # Move tensors to device
            inputs, time_step, qeds, sas = inputs.to(device), time_step.to(device), qeds.to(device), sas.to(device)
            noised_molecule = inputs.reshape(-1, max_seq_length, d_model)
            
            mean_pool = noised_molecule.mean(dim=1)  # Mean pooling
            max_pool = noised_molecule.max(dim=1)[0]  # Max pooling
            first_token_last_layer = noised_molecule[:, 0, :]  # First token output of the last layer

            # Concatenate the vectors
            token_representations = torch.cat([mean_pool, max_pool, first_token_last_layer], dim=1)

            # Forward pass
            qed_outputs = qed_model(token_representations, time_step)
            sas_outputs = sas_model(token_representations, time_step)


            qed_loss = criterion(qed_outputs.squeeze(), qeds)
            sas_loss = criterion(sas_outputs.squeeze(), sas)

            val_qed_loss += qed_loss.item() * inputs.size(0)
            val_sas_loss += sas_loss.item() * inputs.size(0)

    val_qed_loss /= len(val_loader.dataset)
    val_sas_loss /= len(val_loader.dataset)

    print(f'Epoch [{epoch}], '
          f'Train QED Loss: {train_qed_loss:.4f}, Val QED Loss: {val_qed_loss:.4f}, '
          f'Train SAS Loss: {train_sas_loss:.4f}, Val SAS Loss: {val_sas_loss:.4f}')

    # Check early stopping condition
    avg_val_loss = (val_qed_loss + val_sas_loss) / 2
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
    else:
        patience_counter += 1

# Save models
torch.save(qed_model.state_dict(), 'models/qed_model.pt')
torch.save(sas_model.state_dict(), 'models/sas_model.pt')