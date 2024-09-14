import torch
import torch.nn as nn
import torch.optim as optim
from transformers import EsmModel, EsmTokenizer
from torch.utils.data import Dataset, DataLoader, random_split
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import pandas as pd
from tqdm import tqdm
import os
RDLogger.DisableLog('rdApp.*')
from ESM2Regressor import *

# Function to convert SMILES to Morgan fingerprint
def smiles_to_morgan(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return torch.tensor(fingerprint, dtype=torch.float32)

# Custom Dataset class to handle SMILES, protein sequence, and K_d
class ProteinLigandDataset(Dataset):
    def __init__(self, csv_file):
        data = pd.read_csv(csv_file)
        self.sequences = data['protein_sequence']
        self.smiles = data['SMILES']
        self.kd_values = data['k_d'].apply(lambda x: torch.log(torch.tensor(x)))  # Log-transform K_d
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        protein_seq = self.sequences[idx]
        smiles = self.smiles[idx]
        kd_value = self.kd_values[idx]
        
        # Convert SMILES to Morgan fingerprint
        morgan_fingerprint = smiles_to_morgan(smiles)
        
        return protein_seq, morgan_fingerprint, torch.tensor(kd_value, dtype=torch.float32)

# Training and validation function with TQDM
def train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device='cuda', save_path='esm2_regressor.pth'):
    model.to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 5

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        # Training loop with TQDM
        for protein_sequences, morgan_fingerprints, kd_targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            protein_sequences = [seq for seq in protein_sequences]  # List of sequences
            morgan_fingerprints = morgan_fingerprints.to(device)    # Morgan fingerprints
            kd_targets = kd_targets.to(device).float()  # K_d targets
            
            optimizer.zero_grad()
            outputs = model(protein_sequences, morgan_fingerprints).squeeze()
            loss = criterion(outputs, kd_targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for protein_sequences, morgan_fingerprints, kd_targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                protein_sequences = [seq for seq in protein_sequences]
                morgan_fingerprints = morgan_fingerprints.to(device)
                kd_targets = kd_targets.to(device).float()
                
                outputs = model(protein_sequences, morgan_fingerprints).squeeze()
                loss = criterion(outputs, kd_targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if patience_counter > patience:
            break

        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {val_loss:.4f}")
    
    # Save the model
    # torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# Sample data and setup
csv_file = 'data/cleaned_binding_affinities_dataset.csv'
dataset = ProteinLigandDataset(csv_file)

# Split dataset into train and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the model, loss, and optimizer
model = ESM2Regressor()
# model.load_state_dict(torch.load('esm2_regressor.pth', map_location="cuda"))
model.to("cuda")
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train and validate the model
train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50)
