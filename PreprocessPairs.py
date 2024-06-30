import pandas as pd
import tqdm
from MolTransformer import *
from MolLoader import *
from ProteinLoader import *
from ProteinTransformer import *

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

df = pd.read_csv('../data/protein_drug_pairs_with_sequences_and_smiles.csv')
new_df = pd.DataFrame(columns=['Protein CLS', 'SMILES Output', 'pIC50'])

# Load tokenizers
protein_tokenizer = Tokenizer.from_file('../models/protein_tokenizer.json')
protein_max_length = 512
smiles_tokenizer = Tokenizer.from_file('../models/smiles_tokenizer.json')
smiles_max_length = 128

# Load models
protein_vocab_size = 1000
protein_d_model = 768
protein_n_heads = 8
protein_d_ff = 512
protein_n_layers = 5
protein_model = BERT(protein_vocab_size, protein_d_model, protein_max_length, protein_n_heads, protein_d_ff, protein_n_layers)
protein_model.load_state_dict(torch.load('../data/protein_model.pth'))

smiles_vocab_size = 1000
smiles_d_model = 128
smiles_num_heads = 8
smiles_num_layers = 6
smiles_d_ff = 2048
smiles_model = MultiTaskTransformer(smiles_vocab_size, smiles_vocab_size, smiles_d_model, smiles_num_heads, smiles_num_layers, smiles_d_ff, smiles_max_length, 0.1, 1)
smiles_model.load_state_dict(torch.load('../data/smiles_model.pth'))

# loop through entire dataframe row by row
for index, row in df.iterrows():
    protein_sequence = row['Protein Sequence']
    smiles_string = row['SMILES String']
    pIC50 = row['pIC50']
    # preprocess protein sequence
    tokenized_protein = preprocess_protein(protein_tokenizer, protein_sequence, protein_max_length)
    # preprocess SMILES string
    tokenized_smiles = preprocess_smiles(smiles_tokenizer, smiles_string, smiles_max_length)
    # run through transformers
    encoded_protein = torch.tensor(tokenized_protein).unsqueeze(0)
    encoded_smiles = torch.tensor(tokenized_smiles).unsqueeze(0)
    protein_output = protein_model(encoded_protein, torch.ones(encoded_protein.shape)) 
    protein_cls = protein_output[0][0] # get CLS token representation
    smiles_output = smiles_model.encode_smiles(encoded_smiles)

    # save protein_cls, smiles_output, and pIC50 to a new dataframe
    new_df = new_df.append({'Protein CLS': protein_cls, 'SMILES Output': smiles_output, 'pIC50': pIC50}, ignore_index=True)

new_df.to_csv('../data/processed_protein_smiles.csv', index=False)