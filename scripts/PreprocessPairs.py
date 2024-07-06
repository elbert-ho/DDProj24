import pandas as pd
import tqdm
import torch
from tokenizers import Tokenizer
import gc
import numpy as np
from rdkit import Chem
from rdkit.Chem import QED
sys.path.append(os.path.join(os.environ['CONDA_PREFIX'],'share','RDKit','Contrib'))
from SA_Score import sascorer
from MolTransformer import MultiTaskTransformer
from ProteinTransformer import BERT


def preprocess_protein(tokenizer, seq, max_length):
    encoded = tokenizer.encode("[CLS] " + seq + " [EOS]")
    ids = encoded.ids + [tokenizer.token_to_id("[PAD]")] * (max_length - len(encoded.ids))
    ids = ids[:max_length]  # Ensure length does not exceed max_length
    return ids

def preprocess_smiles(tokenizer, smiles, max_length):
    encoded = tokenizer.encode("[CLS] " + smiles + " [EOS]")
    ids = encoded.ids + [tokenizer.token_to_id("[PAD]")] * (max_length - len(encoded.ids))
    ids = ids[:max_length]  # Ensure length does not exceed max_length
    return ids

df = pd.read_csv('data/protein_drug_pairs_with_sequences_and_smiles.csv')
protein_tokenizer = Tokenizer.from_file('models/protein_tokenizer.json')
protein_max_length = 512
smiles_tokenizer = Tokenizer.from_file('models/smiles_tokenizer.json')
smiles_max_length = 128

# Load models
protein_vocab_size = 1000
protein_d_model = 768
protein_n_heads = 8
protein_d_ff = 512
protein_n_layers = 5
protein_model = BERT(protein_vocab_size, protein_d_model, protein_max_length, protein_n_heads, protein_d_ff, protein_n_layers)
protein_model.load_state_dict(torch.load('models/protein_model.pth', map_location='cuda'))
protein_model.to('cuda')

smiles_vocab_size = 1000
smiles_d_model = 128
smiles_num_heads = 8
smiles_num_layers = 6
smiles_d_ff = 2048
smiles_model = MultiTaskTransformer(smiles_vocab_size, smiles_vocab_size, smiles_d_model, smiles_num_heads, smiles_num_layers, smiles_d_ff, smiles_max_length, 0.1, 1)
smiles_model.load_state_dict(torch.load('models/smiles_model.pth', map_location='cuda'))
smiles_model.to('cuda')

# Collect results in a list first
results = []
qed_values = []
sas_values = []

for index, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    protein_sequence = row['Protein Sequence']
    smiles_string = row['SMILES String']
    pIC50 = row['pIC50']

    # Preprocess sequences
    tokenized_protein = preprocess_protein(protein_tokenizer, protein_sequence, protein_max_length)
    tokenized_smiles = preprocess_smiles(smiles_tokenizer, smiles_string, smiles_max_length)

    # Run through transformers
    encoded_protein = torch.tensor(tokenized_protein).unsqueeze(0).to('cuda')
    encoded_smiles = torch.tensor(tokenized_smiles).unsqueeze(0).to('cuda')

    protein_output = protein_model.encode_protein(encoded_protein, torch.ones(encoded_protein.shape).to('cuda'))
    # protein_cls = protein_output[0][0].flatten().cpu().detach().numpy()
    smiles_output = smiles_model.encode_smiles(encoded_smiles)
    smiles_full = smiles_output[0].flatten().cpu().detach().numpy()
    smiles_fingerprint = smiles_output[1].flatten().cpu().detach().numpy()

    # Calculate QED and SAS
    mol = Chem.MolFromSmiles(smiles_string)
    if mol:
        qed_value = QED.qed(mol)
        sas_value = sascorer.calculateScore(mol)
    else:
        qed_value = None
        sas_value = None

    # Collect result
    results.append({'Protein CLS': protein_output, 'SMILES Output': smiles_output, 'SMILES Fingerprint': smiles_fingerprint, 'pIC50': pIC50})
    qed_values.append(qed_value)
    sas_values.append(sas_value)

    # Clean up
    del encoded_protein, encoded_smiles, protein_output, smiles_output
    gc.collect()

# Save the results in a lossless format using np.save
protein_cls_array = np.array([result['Protein CLS'] for result in results])
smiles_output_array = np.array([result['SMILES Output'] for result in results])
smiles_fingerprint_array = np.array([result['SMILES Fingerprint'] for result in results])
pIC50_array = np.array([result['pIC50'] for result in results])
qed_array = np.array(qed_values)
sas_array = np.array(sas_values)

np.save('data/protein_cls.npy', protein_cls_array)
np.save('data/smiles_output.npy', smiles_output_array)
np.save('data/smiles_fingerprint.npy', smiles_output_array)
np.save('data/pIC50.npy', pIC50_array)
np.save('data/qed.npy', qed_array)
np.save('data/sas.npy', sas_array)
