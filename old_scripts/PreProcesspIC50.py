import torch 
from pIC50Predictor import pIC50Predictor
import numpy as np
import yaml

with open("hyperparams.yaml", "r") as file:
    config = yaml.safe_load(file)

proteins = np.load('data/protein_embeddings.npy')
molecules = np.load('data/smiles_fingerprint.npy')

num_diffusion_steps = config["diffusion_model"]["num_diffusion_steps"]
# molecule_dim = config["mol_model"]["d_model"] * config["mol_model"]["max_seq_length"]
molecule_dim = config["mol_model"]["d_model"] * 3

protein_dim = config["protein_model"]["protein_embedding_dim"]
hidden_dim = config["pIC50_model"]["hidden_dim"]
num_heads = config["pIC50_model"]["num_heads"]
lr = config["pIC50_model"]["lr"]
num_epochs = config["pIC50_model"]["num_epochs"]
patience = config["pIC50_model"]["patience"]
time_embed_dim = config["pIC50_model"]["time_embed_dim"]

# Instantiate model, criterion, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = pIC50Predictor(molecule_dim, protein_dim, hidden_dim, num_heads, time_embed_dim, num_diffusion_steps)
model.to(device)
model.load_state_dict(torch.load('models/pIC50_model.pt'))

pIC50List = []
for x in range(len(proteins)):
    for t in range(num_diffusion_steps):
        protein = torch.FloatTensor(proteins[x]).flatten().unsqueeze(0).to(device)
        molecule = torch.FloatTensor(molecules[x]).flatten().unsqueeze(0).to(device)
        time = torch.FloatTensor([t]).to(device)
        pIC50List.append(model(molecule, protein, time))

np.save('data/pIC50.npy', pIC50List)