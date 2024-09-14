# Get ligand 085
import torch
import yaml
from DiffusionModelGLIDE import *
from tqdm import tqdm
from unet_condition import Text2ImUNet
from transformers import EsmTokenizer, EsmModel
from MolTransformerSelfies import MultiTaskTransformer
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem, Descriptors, QED
from rdkit.DataStructs import FingerprintSimilarity
from SelfiesTok import SelfiesTok
import numpy as np
import pandas as pd
import random
import umap
import matplotlib.pyplot as plt
from TestingUtils import *
import sys
import os
sys.path.append(os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'share', 'RDKit', 'Contrib', 'SA_Score'))
import sascorer
import selfies as sf
import argparse

parser = argparse.ArgumentParser(description="Generate molecules with specified protein and parameters.")
args = parser.parse_args()

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
mol_model.load_state_dict(torch.load('models/selfies_transformer_final.pt', map_location=device))
mol_model.to(device)

suppl = Chem.SDMolSupplier('ligands_cl3/ligand085.sdf')
mol = suppl[0]
smiles = Chem.MolToSmiles(mol)
selfies_toks = sf.split_selfies(sf.encoder(smiles))
tokenizer = SelfiesTok.load("models/selfies_tok.json")

encoded = tokenizer.encode(["[CLS]"] + list(selfies_toks) + ["[EOS]"])
# print(self.tokenizer.token_to_id)
# print(selfies_toks)
# print(encoded)
ids = encoded + [tokenizer.token_to_id["[PAD]"]] * (max_seq_length - len(encoded))
ids = ids[:max_seq_length]  # Ensure length does not exceed max_length
ids_final = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
encoded = mol_model.encode_smiles(ids_final)[0].flatten().detach().cpu().numpy()

smiles_maxes = np.load("data/smiles_maxes_selfies.npy")
smiles_mins = np.load("data/smiles_mins_selfies.npy")
# smiles_maxes = np.amax(smiles, axis=0).reshape(1, -1)
# smiles_mins = np.amin(smiles, axis=0).reshape(1, -1)
encoded = 2 * (encoded - smiles_mins) / (smiles_maxes - smiles_mins) - 1
# print(encoded.shape)
# exit()
np.save("data/ref_sample.npy", encoded)