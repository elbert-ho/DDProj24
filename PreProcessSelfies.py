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
import yaml
import selfies as sf
from SelfiesTok import SelfiesTok
# Suppress warnings
RDLogger.DisableLog('rdApp.*')
ESM = False
smiles_only = True

def save_batch(data, filename):
    np.save(filename, np.array(data))

if ESM:
# Load ESM-2 model for protein processing
    protein_model_name = "facebook/esm2_t30_150M_UR50D"
    protein_tokenizer = EsmTokenizer.from_pretrained(protein_model_name)
    protein_model = EsmModel.from_pretrained(protein_model_name).to('cuda')

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

smiles_model = MultiTaskTransformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, num_tasks)
smiles_model.load_state_dict(torch.load('models/selfies_transformer_final.pt'))
# Ensure the model is on the correct device
smiles_model.to(device)
tokenizer = SelfiesTok.load("models/selfies_tok.json")
# Initialize the dataset and dataloaders
smiles_model.eval()

batch_size = 1000
num_batches = len(df) // batch_size + int(len(df) % batch_size > 0)

for batch_index in tqdm.tqdm(range(num_batches)):
    start_index = batch_index * batch_size
    end_index = min((batch_index + 1) * batch_size, len(df))
    batch_df = df[start_index:end_index]

    if ESM:
        protein_embeddings_batch = []
    smiles_output_batch = []
    smiles_fingerprint_batch = []

    if not smiles_only:
        pIC50_batch = []
        qed_batch = []
        sas_batch = []

    for index, row in tqdm.tqdm(batch_df.iterrows(), total=len(batch_df)):
        if ESM:
            protein_sequence = row['Protein Sequence']
        smiles_string = row['SMILES String']
        
        if not smiles_only:
            pIC50 = row['pIC50']

        if ESM:
            encoded_protein = protein_tokenizer(protein_sequence, return_tensors='pt', padding=True, truncation=True).to('cuda')
            # Generate protein embeddings
            with torch.no_grad():
                protein_outputs = protein_model(**encoded_protein)
                protein_embeddings = protein_outputs.last_hidden_state

                # Mean and Max Pooling
                mean_pooled = protein_embeddings.mean(dim=1)
                max_pooled = protein_embeddings.max(dim=1).values
                combined_pooled = torch.cat((mean_pooled, max_pooled), dim=1)

        # Preprocess and encode 
        
        selfies_toks = sf.split_selfies(sf.encoder(smiles_string))
        selfies_toks = tokenizer.encode(["[CLS]"] + list(selfies_toks) + ["[EOS]"])
        ids = selfies_toks + [tokenizer.token_to_id["[PAD]"]] * (max_seq_length - len(selfies_toks))
        ids = ids[:max_seq_length] 
        tokenized_smiles = ids
        encoded_smiles = torch.tensor(tokenized_smiles).unsqueeze(0).to('cuda')
        smiles_output = smiles_model.encode_smiles(encoded_smiles)
        smiles_full = smiles_output[0].flatten().cpu().detach().numpy()
        smiles_fingerprint = smiles_output[1].flatten().cpu().detach().numpy()

        if not smiles_only:
            # Calculate QED and SAS
            mol = Chem.MolFromSmiles(smiles_string)
            if mol:
                qed_value = QED.qed(mol)
                sas_value = sascorer.calculateScore(mol)
            else:
                qed_value = None
                sas_value = None

        # Collect result
        if ESM:
            protein_embeddings_batch.append(combined_pooled.cpu().numpy())
        smiles_output_batch.append(smiles_full)
        smiles_fingerprint_batch.append(smiles_fingerprint)
        
        if not smiles_only:
            pIC50_batch.append(pIC50)
            qed_batch.append(qed_value)
            sas_batch.append(sas_value)

        # Clean up
        if ESM:
            del protein_outputs, encoded_protein
        del encoded_smiles, smiles_output
        torch.cuda.empty_cache()
        gc.collect()

    # Save the batch results
    if ESM:
        save_batch(protein_embeddings_batch, f'data/protein_embeddings_batch_{batch_index}.npy')
    
    save_batch(smiles_output_batch, f'data/smiles_output_batch_{batch_index}.npy')
    save_batch(smiles_fingerprint_batch, f'data/smiles_fingerprint_batch_{batch_index}.npy')
    
    if not smiles_only:
        save_batch(pIC50_batch, f'data/pIC50_batch_{batch_index}.npy')
        save_batch(qed_batch, f'data/qed_batch_{batch_index}.npy')
        save_batch(sas_batch, f'data/sas_batch_{batch_index}.npy')

# Concatenate all the saved batches
def load_batches(pattern):
    batch_files = sorted([f for f in os.listdir('data') if f.startswith(pattern)])
    return np.concatenate([np.load(os.path.join('data', f)) for f in batch_files])

if ESM:
    protein_embeddings_array = load_batches('protein_embeddings_batch_')
    np.save('data/protein_embeddings.npy', protein_embeddings_array)

smiles_output_array = load_batches('smiles_output_batch_')
smiles_fingerprint_array = load_batches('smiles_fingerprint_batch_')

if not smiles_only:
    pIC50_array = load_batches('pIC50_batch_')
    qed_array = load_batches('qed_batch_')
    sas_array = load_batches('sas_batch_')

# Save the concatenated results
np.save('data/smiles_output_selfies.npy', smiles_output_array)
np.save('data/smiles_fingerprint_selfies.npy', smiles_fingerprint_array)

if not smiles_only:
    np.save('data/pIC50.npy', pIC50_array)
    np.save('data/qed.npy', qed_array)
    np.save('data/sas.npy', sas_array)

for i in range(num_batches):
    os.remove(f'data/smiles_output_batch_{i}.npy')
    os.remove(f'data/smiles_fingerprint_batch_{i}.npy')