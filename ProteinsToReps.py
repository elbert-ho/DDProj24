import pandas as pd
import torch
from ESM2Regressor import ESM2Regressor
import numpy as np

df = pd.read_csv('data/protein_drug_pairs_with_sequences_and_smiles_cleaned.csv')
# df = pd.read_csv('data/protein_drug_pairs_with_sequences_and_smiles_cleaned2.csv')

# proteins = np.load("data/protein_embeddings3.npy")
# proteins = np.load("data/protein_embeddings4.npy")

proteins = df['Protein Sequence']

model = ESM2Regressor()
model.load_state_dict(torch.load('esm2_regressor_saved.pth', map_location="cuda"))
model.to("cuda")

representations = []
for protein_sequence in proteins:
    representation = model.get_rep(protein_sequence).flatten().detach().cpu().numpy()
    representations.append(representation)

np.save('data/protein_representations.npy', representations)