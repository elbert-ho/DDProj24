import torch
import torch.nn as nn
import torch.nn.functional as F
from SinPosEmb import getTimeMLP

class pIC50Predictor(nn.Module):
    def __init__(self, molecule_dim, protein_dim, num_heads, embedding_dim, max_timesteps):
        super(pIC50Predictor, self).__init__()
        # Embedding layers
        self.molecule_fc1 = nn.Linear(molecule_dim + embedding_dim, 4096)
        self.mbn1 = nn.LayerNorm(4096)
        self.molecule_fc2 = nn.Linear(4096, 2048)
        self.mbn2 = nn.LayerNorm(2048)
        self.molecule_fc3 = nn.Linear(2048, 1024) 
        self.mbn3 = nn.LayerNorm(1024)
        self.protein_fc = nn.Linear(protein_dim + embedding_dim, 1024)
        self.pbn1 = nn.LayerNorm(1024)
        self.posEmb1 = getTimeMLP(embedding_dim / 4, max_timesteps, embedding_dim)
        self.posEmb2 = getTimeMLP(embedding_dim / 4, max_timesteps, embedding_dim)
        self.posEmb3 = getTimeMLP(embedding_dim / 4, max_timesteps, embedding_dim)

        self.fc_connector = nn.Linear(2048 + embedding_dim, 1024)
        self.ln_connector = nn.LayerNorm(1024)
        # Cross-attention layers
        # self.cross_attention = nn.MultiheadAttention(1024, num_heads)

        # Fully connected layers
        self.fc1 = nn.Linear(1024 + embedding_dim, 1024)
        self.bn1 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.LayerNorm(256)
        self.fc4 = nn.Linear(256, 1)

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, molecule_repr, protein_repr, time_step):
        device = molecule_repr.device
        self.to(device)

        protein_repr = protein_repr.squeeze(1)
        # Process molecule and protein representations
        time1 = self.posEmb1(time_step).to(device)
        time2 = self.posEmb2(time_step).to(device)
        time3 = self.posEmb3(time_step).squeeze(1).to(device)

        # Concatenate time embeddings instead of adding
        mol_hidden = torch.cat((molecule_repr, time1), dim=-1)
        mol_hidden = self.dropout(F.relu(self.mbn1(self.molecule_fc1(mol_hidden))))
        mol_hidden = self.dropout(F.relu(self.mbn2(self.molecule_fc2(mol_hidden))))
        mol_hidden = self.dropout(F.relu(self.mbn3(self.molecule_fc3(mol_hidden))))
        
        prot_hidden = torch.cat((protein_repr, time2), dim=-1)
        prot_hidden = self.dropout(F.relu(self.pbn1(self.protein_fc(prot_hidden))))

        # Prepare for cross-attention
        # mol_hidden = mol_hidden.unsqueeze(1)  # (batch_size, hidden_dim) -> (batch_size, 1, hidden_dim)
        # prot_hidden = prot_hidden.unsqueeze(1)  # (batch_size, hidden_dim) -> (batch_size, 1, hidden_dim)

        # # Transpose to (seq_len, batch_size, hidden_dim) as required by MultiheadAttention
        # mol_hidden = mol_hidden.transpose(0, 1)  # (1, batch_size, hidden_dim)
        # prot_hidden = prot_hidden.transpose(0, 1)  # (1, batch_size, hidden_dim)

        # Cross-attention mechanism
        # attention_output, _ = self.cross_attention(mol_hidden, prot_hidden, prot_hidden)
        # attention_output = attention_output.transpose(0, 1).squeeze(1)  # (1, batch_size, hidden_dim) -> (batch_size, hidden_dim)

        # Concatenate attention output with time embedding
        print(time3.shape)
        combined_repr = torch.cat((mol_hidden, prot_hidden, time3), dim=-1)
        print(combined_repr.shape)
        hidden = self.dropout(F.relu(self.ln_connector(self.fc_connector(combined_repr))))
        # Fully connected layers with batch norm and dropout
        hidden = self.dropout(F.relu(self.bn1(self.fc1(hidden))))
        hidden = self.dropout(F.relu(self.bn2(self.fc2(hidden))))
        hidden = self.dropout(F.relu(self.bn3(self.fc3(hidden))))
        pIC50 = self.fc4(hidden)

        return pIC50
