import os
import pandas as pd
import tqdm
import torch
from tokenizers import Tokenizer
import gc
import warnings
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import QED
from transformers import EsmTokenizer, EsmModel
import sys
sys.path.append(os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'share', 'RDKit', 'Contrib', 'SA_Score'))
import sascorer
from MolTransformerSelfies import MultiTaskTransformer
from MolLoaderSelfiesFinal import SMILESDataset
import yaml
import selfies as sf


with open("hyperparams.yaml", "r") as file:
    config = yaml.safe_load(file)

file_path = "data/pd_truncated_final_1.csv"
df = pd.read_csv(file_path)
dataset = SMILESDataset(file_path, tokenizer_path="models/selfies_tokenizer_final.json", unicode_path="models/unicode_mapping.json", props=False)

src_vocab_size = config["mol_model"]["src_vocab_size"]
tgt_vocab_size = config["mol_model"]["tgt_vocab_size"]
max_seq_length = config["mol_model"]["max_seq_length"]
num_tasks = config["mol_model"]["num_tasks"]
d_model = config["mol_model"]["d_model"]
num_heads = config["mol_model"]["num_heads"]
num_layers = config["mol_model"]["num_layers"]
d_ff = config["mol_model"]["d_ff"]
dropout = config["mol_model"]["dropout"]
learning_rate = config["mol_model"]["learning_rate"]
batch_size = config["mol_model"]["batch_size"]
device = config["mol_model"]["device"]
warmup_epochs = config["mol_model"]["warmup_epochs"]
total_epochs = config["mol_model"]["total_epochs"]
patience = config["mol_model"]["patience"]
pretrain_epochs = config["mol_model"]["pretrain_epochs"]
pretrain_learning_rate = config["mol_model"]["pretrain_learning_rate"]
# tok_file = config["mol_model"]["tokenizer_file"]

smiles_model = MultiTaskTransformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, num_tasks)
smiles_model.load_state_dict(torch.load('models/selfies_transformer_final_bpe.pt'))
# Ensure the model is on the correct device
smiles_model.to(device)
# tokenizer = SelfiesTok.load("models/selfies_tok.json")
# Initialize the dataset and dataloaders
smiles_model.eval()

selfies_toks = (sf.encoder("[C@@H1]([C@@H1]([NH1])[C@@H1]CC1=C2C=C(O)C=C1)=CC3[C@@H1]C=C(O)C=C3S[NH1]2"))
selfies_toks = dataset.encode(selfies_toks)
ids = selfies_toks + [dataset.tokenizer.token_to_id("[PAD]")] * (max_seq_length - len(selfies_toks))
ids = ids[:max_seq_length] 
tokenized_smiles = ids
encoded_smiles = torch.tensor(tokenized_smiles).unsqueeze(0).to('cuda')
smiles_output_here = smiles_model.encode_smiles(encoded_smiles)
smiles_full = smiles_output_here[0].reshape(1, 32768).cpu().detach().numpy()
# smiles_output.append(smiles_full)

print(smiles_full.shape)

np.save('data/mol_19.npy', smiles_full)