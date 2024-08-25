# IMPORTS
import torch
import yaml
from DiffusionModelGLIDE import *
from tqdm import tqdm
from unet_condition import Text2ImUNet
from transformers import EsmTokenizer, EsmModel
from MolTransformerSelfies import MultiTaskTransformer
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from fcd_torch import FCD
from SelfiesTok import SelfiesTok
import numpy as np
import pandas as pd
import random
import umap
import matplotlib.pyplot as plt
from typing import Set, Any
from TestingUtils import *
from scipy.stats import wasserstein_distance
import sys
import os
sys.path.append(os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'share', 'RDKit', 'Contrib', 'SA_Score'))
import sascorer
from rdkit.Chem import Descriptors, QED
from collections import Counter
import selfies as sf
from pIC50Predictor2 import pIC50Predictor
import gc

# Step 0: Load and generate
# Step 0.1: Load in the training data (potential source of issue as this includes val but whatever)
ref_file_smiles = "data/protein_drug_pairs_with_sequences_and_smiles.csv"
ref_file_finger = "data/smiles_output_selfies.npy"
ref_data_finger = np.load(ref_file_finger)
ref_data_smiles = pd.read_csv(ref_file_smiles)

# Step 0.2: Load in GaussianDiffusion model and unet model and MolTransformer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open("hyperparams.yaml", "r") as file:
    config = yaml.safe_load(file)

n_diff_step = config["diffusion_model"]["num_diffusion_steps"]
protein_embedding_dim = config["protein_model"]["protein_embedding_dim"]
batch_size = config["diffusion_model"]["batch_size"]
src_vocab_size = config["mol_model"]["src_vocab_size"]
tgt_vocab_size = config["mol_model"]["tgt_vocab_size"]
max_seq_length = config["mol_model"]["max_seq_length"]
num_tasks = config["mol_model"]["num_tasks"]
d_model = config["mol_model"]["d_model"]
num_heads = config["mol_model"]["num_heads"]
num_layers = config["mol_model"]["num_layers"]
d_ff = config["mol_model"]["d_ff"]
dropout = config["mol_model"]["dropout"]

diffusion_model = GaussianDiffusion(betas=get_named_beta_schedule(n_diff_step))
unet = Text2ImUNet(text_ctx=1, xf_width=protein_embedding_dim, xf_layers=0, xf_heads=0, xf_final_ln=0, tokenizer=None, in_channels=256, model_channels=256, out_channels=512, num_res_blocks=2, attention_resolutions=[4], dropout=.1, channel_mult=(1, 2, 4, 8), dims=1)
mol_model = MultiTaskTransformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, num_tasks).to(device)

unet.load_state_dict(torch.load('unet_resized_even_attention_4_tuned.pt', map_location=device))
mol_model.load_state_dict(torch.load('models/selfies_transformer_final.pt', map_location=device))

unet, mol_model = unet.to(device), mol_model.to(device)
mol_model.eval()

# Step 0.3: Pick 40 random proteins and generate 25 molecules per protein
# Group by Protein Sequence and filter groups that have at least 25 SMILES strings
ref_data_smiles.reset_index(inplace=True)
protein_groups = ref_data_smiles.groupby('Protein Sequence').filter(lambda x: len(x) >= 50)

# Get unique protein sequences that have at least 25 SMILES strings
unique_proteins = protein_groups['Protein Sequence'].unique()

# Select 40 random unique proteins HERE
num_proteins = 20
per_protein = 50

random_proteins = random.sample(list(unique_proteins), num_proteins)

# For each selected protein, extract 25 random SMILES strings
selected_rows = []
for protein in random_proteins:
    protein_df = protein_groups[protein_groups['Protein Sequence'] == protein]
    selected_smiles = protein_df.sample(per_protein)
    selected_rows.append(selected_smiles)

ref_data_smiles_cut = pd.concat(selected_rows)

# Get ESM
protein_indices = ref_data_smiles_cut.loc[:,"index"].to_numpy()
protein_fingers_full = np.load("data/protein_embeddings.npy")
protein_fingers_cut = []
count = 0
for protein_idx in protein_indices:
    if count % per_protein == 0:
        protein_fingers_cut.append(protein_fingers_full[protein_idx])
    count += 1

# print(len(protein_fingers_cut))
# def get_pIC50_grad(x, prot, t):
#     pic50_model.train()
#     x = x.detach().requires_grad_(True)
#     # print(x)
#     # print(prot)
#     # print(t)
#     # print(pic50_model.training)
#     pic50_pred = pic50_model(x, prot, t)
#     # gradients = []
#     # for i in range(x.shape[0]):
#     #     grad = torch.autograd.grad(outputs=pic50_pred[i], inputs=x[i])[0].deatch()
#     #     gradients.append(grad)
#     # grad = torch.stack(gradients)
#     # print(pic50_pred.requires_grad)
#     grad = torch.autograd.grad(pic50_pred.sum(), x)[0].detach()
#     return grad

# Step 0.3.1: Generate 25 molecules per protein
mins = torch.tensor(np.load("data/smiles_mins_selfies.npy"), device=device).reshape(1, 1, -1)
maxes = torch.tensor(np.load("data/smiles_maxes_selfies.npy"), device=device).reshape(1, 1, -1)
ws = [1, 3, 5, 7, 9]
fcd_scores = []
fcd_scores_full = []
for w in ws:
    samples = torch.tensor([], device=device)
    for protein_finger in protein_fingers_cut:
        protein_finger = torch.tensor(protein_finger, device=device)
        protein_finger = protein_finger.repeat(per_protein, 1)
        sample = diffusion_model.p_sample_loop(unet, (per_protein, 256, 128), prot=protein_finger, w=w).detach().reshape(per_protein, 1, 32768)
        # sample = diffusion_model.p_sample_loop(unet, (10, 256, 128), prot=protein_finger, cond_fn=get_pIC50_grad).detach().reshape(per_protein, 1, 32768)
        samples = torch.cat([samples, sample])
    # Debug line
    # print(samples.shape)

    # Step 0.3.2: Remember to unnormalize
    sample_rescale = (((samples + 1) / 2) * (maxes - mins) + mins)
    # print(sample_rescale.shape)

    # Step 0.3.3: Convert from SELFIES back to SMILES
    tokenizer = SelfiesTok.load("models/selfies_tok.json")
    with torch.no_grad():
        decoded_smiles, _ = mol_model.decode_representation(sample_rescale.reshape(-1, max_seq_length, d_model), None, max_length=128, tokenizer=tokenizer)

    gen_smiles = []

    for decode in decoded_smiles:
        predicted_selfie = tokenizer.decode(decode.detach().cpu().flatten().tolist(), skip_special_tokens=True)
        predicted_smile = sf.decoder(predicted_selfie)
        gen_smiles.append(predicted_smile)

    # Assuming protein_indices and sample_rescale are already defined elsewhere in your code
    smiles_fingers_cut = []
    smiles_fingers_full = np.load("data/smiles_output_selfies.npy")
    for protein_idx in protein_indices:
        smiles_fingers_cut.append(smiles_fingers_full[protein_idx].tolist())

    # Convert to numpy array
    smiles_fingers_cut = np.array(smiles_fingers_cut)
    gen_smiles = np.array(gen_smiles)

    gen_mols = []
    valid = 0
    total = 0
    for gen_smile in gen_smiles:
        try:
            mol = Chem.MolFromSmiles(gen_smile)
            valid += 1
        except:
            pass
        gen_mols.append(mol)
        total += 1

    ref_mols = []
    ref_smiles = []
    # ref_data_smiles_cut
    for smiles in ref_data_smiles_cut.loc[:,"SMILES String"]:
        ref_smiles.append(smiles)
        mol = Chem.MolFromSmiles(smiles)
        ref_mols.append(mol)

    with torch.no_grad():
        fcd = FCD(device='cuda:0', n_jobs=1)
        # fcd_score = 0
        # for idx in range(num_proteins):
            # fcd_score += fcd(gen_smiles[50 * (idx): 50 * (idx + 1)], ref_smiles[50 * (idx): 50 * (idx + 1)])

        # fcd_score /= num_proteins
        # fcd_scores.append(fcd_score)
        fcd_score = fcd(gen_smiles, ref_smiles)
        fcd_scores_full.append(fcd_score)
        print(f"w: {w} - fcd: {fcd_score}")

    gc.collect()

# plt.plot(ws, fcd_scores, 'o-', label='FCD Average vs. w')
plt.plot(ws, fcd_scores_full, 'o-', label='FCD Full vs. w')
# Adding labels and title
plt.xlabel('w')
plt.ylabel('FCD')
plt.legend()
plt.savefig("fcdw.png")

