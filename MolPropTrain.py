import torch
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from MolPropLoader import MolPropDataset
from MolPropModel import PropertyModel
import torch.nn as nn
from tqdm import tqdm

# Load the dataset
dataset = MolPropDataset(smiles_fingerprint_file='data/smiles_fingerprint.npy', qed_file='data/qed.npy', sas_file = 'data/sas.npy')

# Split into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize models
input_size = dataset[0][0].shape[0]
qed_model = PropertyModel(input_size)
sas_model = PropertyModel(input_size)

# Define loss function and optimizers
criterion = nn.MSELoss()
qed_optimizer = optim.Adam(qed_model.parameters(), lr=0.001)
sas_optimizer = optim.Adam(sas_model.parameters(), lr=0.001)

# Early stopping parameters
patience = 5
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

    for inputs, qeds, sas in tqdm(train_loader, desc=f"Epoch {epoch} [Training]"):
        # Zero the parameter gradients
        qed_optimizer.zero_grad()
        sas_optimizer.zero_grad()

        # Forward pass
        qed_outputs = qed_model(inputs)
        sas_outputs = sas_model(inputs)

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
        for inputs, qeds, sas in tqdm(val_loader, desc=f"Epoch {epoch} [Validation]"):
            qed_outputs = qed_model(inputs)
            sas_outputs = sas_model(inputs)

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
