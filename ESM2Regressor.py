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

# The modified model that accepts both protein and ligand fingerprints
class ESM2Regressor(nn.Module):
    def __init__(self, esm_model_name="facebook/esm2_t6_8M_UR50D", protein_hidden_size=512, dropout=0.1, fingerprint_size=2048):
        super(ESM2Regressor, self).__init__()
        
        # Load the pre-trained ESM2 model and tokenizer from transformers
        self.esm_model = EsmModel.from_pretrained(esm_model_name)
        self.tokenizer = EsmTokenizer.from_pretrained(esm_model_name)
        self.esm_model.eval()  # Set model to evaluation mode since we're extracting features
        
        # Freeze the ESM2 model parameters (protein embedding extraction)
        for param in self.esm_model.parameters():
            param.requires_grad = False
        
        # Dense layers for the protein sequence representation (2-3 layers)
        seq_len = self.esm_model.config.max_position_embeddings - 2
        embedding_dim = self.esm_model.config.hidden_size

        # print(seq_len)
        # print(embedding_dim)
        # exit()

        self.fc_protein1 = nn.Linear(seq_len * embedding_dim, protein_hidden_size)
        self.fc_protein2 = nn.Linear(protein_hidden_size, protein_hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        
        # Final layer combining the protein fingerprint and molecular fingerprint
        combined_input_size = (protein_hidden_size // 2) + fingerprint_size
        self.fc_combined = nn.Linear(combined_input_size, 1)  # Predicting log(K_d)
    
    def forward(self, protein_sequence, morgan_fingerprint):
        # Tokenize and extract protein sequence embeddings
        inputs = self.tokenizer(protein_sequence, return_tensors="pt", padding='max_length', truncation=True, max_length=1024).to(morgan_fingerprint.device)  # Ensure tokenized inputs are on the same device
        with torch.no_grad():
            outputs = self.esm_model(**inputs)
        
        # print(outputs.last_hidden_state.shape)
        

        # Flatten the last_hidden_state
        last_hidden_state = outputs.last_hidden_state
        protein_rep = last_hidden_state.view(last_hidden_state.size(0), -1)  # Flatten
        
        # Pass the protein embedding through 2 dense layers
        x_protein = torch.relu(self.fc_protein1(protein_rep))
        x_protein = self.dropout(x_protein)
        x_protein = torch.relu(self.fc_protein2(x_protein))
        
        # Combine protein and ligand (Morgan fingerprint) representations at the final layer
        combined = torch.cat((x_protein, morgan_fingerprint), dim=1)
        log_kd_pred = self.fc_combined(combined)
        
        return log_kd_pred
    
    def get_rep(self, protein_sequence):
        inputs = self.tokenizer(protein_sequence, return_tensors="pt", padding='max_length', truncation=True, max_length=1024).to("cuda")  # Ensure tokenized inputs are on the same device
        with torch.no_grad():
            outputs = self.esm_model(**inputs)
        
        # print(outputs.last_hidden_state.shape

        # Flatten the last_hidden_state
        last_hidden_state = outputs.last_hidden_state
        protein_rep = last_hidden_state.view(last_hidden_state.size(0), -1)  # Flatten        # Pass the protein embedding through 2 dense layers
        x_protein = torch.relu(self.fc_protein1(protein_rep))
        x_protein = self.dropout(x_protein)
        x_protein = torch.relu(self.fc_protein2(x_protein))
        return x_protein
    