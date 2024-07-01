import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from pIC50Loader import pIC50Dataset
from DiffusionModel import DiffusionModel

class pIC50Predictor(nn.Module):
    def __init__(self, molecule_dim, protein_dim, time_dim, hidden_dim, num_heads):
        super(pIC50Predictor, self).__init__()

        # Embedding layers
        self.molecule_fc = nn.Linear(molecule_dim, hidden_dim)
        self.protein_fc = nn.Linear(protein_dim, hidden_dim)

        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim + time_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

        # Dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, molecule_repr, protein_repr, time_step):
        # Process molecule and protein representations
        mol_hidden = F.relu(self.molecule_fc(molecule_repr))
        prot_hidden = F.relu(self.protein_fc(protein_repr))

        # Debug statements to check shapes
        # print(f"mol_hidden shape after embedding: {mol_hidden.shape}")
        # print(f"prot_hidden shape after embedding: {prot_hidden.shape}")

        # Remove any extra dimensions from mol_hidden
        mol_hidden = mol_hidden.squeeze(1) if len(mol_hidden.shape) > 2 else mol_hidden

        # Prepare for cross-attention
        mol_hidden = mol_hidden.unsqueeze(1)  # (batch_size, hidden_dim) -> (batch_size, 1, hidden_dim)
        prot_hidden = prot_hidden.unsqueeze(1)  # (batch_size, hidden_dim) -> (batch_size, 1, hidden_dim)

        # Debug statements to check shapes
        # print(f"mol_hidden shape after unsqueeze: {mol_hidden.shape}")
        # print(f"prot_hidden shape after unsqueeze: {prot_hidden.shape}")

        # Transpose to (seq_len, batch_size, hidden_dim) as required by MultiheadAttention
        mol_hidden = mol_hidden.transpose(0, 1)  # (1, batch_size, hidden_dim)
        prot_hidden = prot_hidden.transpose(0, 1)  # (1, batch_size, hidden_dim)

        # Debug statements to check shapes
        # print(f"mol_hidden shape after transpose: {mol_hidden.shape}")
        # print(f"prot_hidden shape after transpose: {prot_hidden.shape}")

        # Cross-attention mechanism
        attention_output, _ = self.cross_attention(mol_hidden, prot_hidden, prot_hidden)
        attention_output = attention_output.transpose(0, 1).squeeze(1)  # (1, batch_size, hidden_dim) -> (batch_size, hidden_dim)

        # Debug statements to check shapes
        # print(f"attention_output shape after cross_attention: {attention_output.shape}")

        # Concatenate all representations
        combined_repr = torch.cat((attention_output, time_step), dim=1)

        # Fully connected layers with dropout
        hidden = F.relu(self.fc1(combined_repr))
        hidden = self.dropout(hidden)
        hidden = F.relu(self.fc2(hidden))
        hidden = self.dropout(hidden)
        pIC50 = self.fc3(hidden)

        return pIC50

def train(model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs=100, patience=10):
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(num_epochs)):
        model.train()
        train_loss = 0.0

        for noised_molecule, protein, time_step, pIC50 in train_dataloader:
            noised_molecule = noised_molecule.to(device)
            protein = protein.to(device)
            time_step = time_step.to(device)
            pIC50 = pIC50.to(device)

            optimizer.zero_grad()
            outputs = model(noised_molecule, protein, time_step)
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
                noised_molecule = noised_molecule.to(device)
                protein = protein.to(device)
                time_step = time_step.to(device)
                pIC50 = pIC50.to(device)

                outputs = model(noised_molecule, protein, time_step)
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
# Ensure you have defined the DiffusionModel and pIC50Dataset classes
protein_file = 'protein_cls.npy'
smiles_file = 'smiles_output.npy'
pIC50_file = 'pIC50.npy'
diffusion_model = DiffusionModel(beta_start=0.1, beta_end=0.2, num_diffusion_steps=1000)
dataset = pIC50Dataset(protein_file, smiles_file, pIC50_file, diffusion_model, num_diffusion_steps=1000)

# Train-test split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model parameters
molecule_dim = 16384
protein_dim = 1000
time_dim = 1
hidden_dim = 256
num_heads = 8

# Instantiate model, criterion, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = pIC50Predictor(molecule_dim, protein_dim, time_dim, hidden_dim, num_heads)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
train(model, train_dataloader, val_dataloader, criterion, optimizer, device)
