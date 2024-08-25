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
from MolTransformer import MultiTaskTransformer
import yaml

# Suppress warnings
RDLogger.DisableLog('rdApp.*')

def save_batch(data, filename):
    np.save(filename, np.array(data))

# Load SMILES model
with open("hyperparams.yaml", "r") as file:
    config = yaml.safe_load(file)

df = pd.read_csv('data/protein_drug_pairs_with_sequences_and_smiles.csv')
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
tok_file = config["mol_model"]["tokenizer_file"]


batch_size = 1000
num_batches = len(df) // batch_size + int(len(df) % batch_size > 0)

for batch_index in tqdm.tqdm(range(num_batches)):
    start_index = batch_index * batch_size
    end_index = min((batch_index + 1) * batch_size, len(df))
    batch_df = df[start_index:end_index]

    pIC50_batch = batch_df['pIC50'].to_numpy()
    # Save the batch result
    save_batch(pIC50_batch, f'data/pIC50_batch_{batch_index}.npy')


# Concatenate all the saved batches
def load_batches(pattern):
    batch_files = sorted([f for f in os.listdir('data') if f.startswith(pattern)])
    return np.concatenate([np.load(os.path.join('data', f)) for f in batch_files])

pIC50_array = load_batches('pIC50_batch_')

# Save the concatenated results
np.save('data/pIC50.npy', pIC50_array)