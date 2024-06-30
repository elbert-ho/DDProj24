import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from DiffusionModel import *
from pIC50Loader import *

class pIC50Predictor(nn.Module):
    def __init__(self, molecule_dim, protein_dim, time_dim, hidden_dim, num_heads, num_layers):
        super(pIC50Predictor, self).__init__()
        
        # Embedding layers
        self.molecule_fc = nn.Linear(molecule_dim, hidden_dim)
        self.protein_fc = nn.Linear(protein_dim, hidden_dim)
        
        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads, num_layers)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2 + 1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        
        # Dropout
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, molecule_repr, protein_repr, time_step):
        # Process molecule and protein representations
        mol_hidden = F.relu(self.molecule_fc(molecule_repr))
        prot_hidden = F.relu(self.protein_fc(protein_repr))
        
        # Cross-attention mechanism
        attention_output, _ = self.cross_attention(mol_hidden.unsqueeze(1), prot_hidden.unsqueeze(1), prot_hidden.unsqueeze(1))
        attention_output = attention_output.squeeze(1)
        
        # Concatenate all representations
        combined_repr = torch.cat((attention_output, time_step), dim=1)
        
        # Fully connected layers with dropout
        hidden = F.relu(self.fc1(combined_repr))
        hidden = self.dropout(hidden)
        hidden = F.relu(self.fc2(hidden))
        hidden = self.dropout(hidden)
        pIC50 = self.fc3(hidden)
        
        return pIC50


def train(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs=100, patience=10):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for noised_molecule, protein, time_step, pIC50 in train_dataloader:
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
csv_file = '../data/protein_drug_encoded.csv'
diffusion_model = DiffusionModel(beta_start=0.1, beta_end=0.2, num_diffusion_steps=1000)
dataset = pIC50Dataset(csv_file, diffusion_model, num_diffusion_steps=1000)

# Train-test split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
molecule_dim = 128
protein_dim = 128
time_dim = 1
hidden_dim = 256
num_heads = 8
num_layers = 4

model = pIC50Predictor(molecule_dim, protein_dim, time_dim, hidden_dim, num_heads, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train(model, train_dataloader, val_dataloader, criterion, optimizer)
