import torch
import torch.nn as nn
import torch.nn.functional as F
from SinPosEmb import getTimeMLP

class pIC50Predictor(nn.Module):
    def __init__(self, molecule_dim, protein_dim, hidden_dim, num_heads, embedding_dim, max_timesteps):
        super(pIC50Predictor, self).__init__()

        # Embedding layers
        self.molecule_fc = nn.Linear(molecule_dim, hidden_dim)
        self.protein_fc = nn.Linear(protein_dim, hidden_dim)
        self.positional_embedding = getTimeMLP(embedding_dim / 4, max_timesteps, embedding_dim)

        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim + embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)

        # Dropout
        self.dropout = nn.Dropout(0.2)

    def forward(self, molecule_repr, protein_repr, time_step):
        # Process molecule and protein representations
        mol_hidden = F.relu(self.molecule_fc(molecule_repr))
        prot_hidden = F.relu(self.protein_fc(protein_repr))

        # Debug statements to check shapes
        # print(f"mol_hidden shape after embedding: {mol_hidden.shape}")
        # print(f"prot_hidden shape after embedding: {prot_hidden.shape}")

        # Remove any extra dimensions from mol_hidden
        mol_hidden = mol_hidden.squeeze(1) if len(mol_hidden.shape) > 2 else mol_hidden

        # Prepare for cross-attention
        mol_hidden = mol_hidden.unsqueeze(1)  # (batch_size, hidden_dim) -> (batch_size, 1, hidden_dim)
        prot_hidden = prot_hidden.unsqueeze(1)  # (batch_size, hidden_dim) -> (batch_size, 1, hidden_dim)

        # Debug statements to check shapes
        # print(f"mol_hidden shape after unsqueeze: {mol_hidden.shape}")
        # print(f"prot_hidden shape after unsqueeze: {prot_hidden.shape}")

        # Transpose to (seq_len, batch_size, hidden_dim) as required by MultiheadAttention
        mol_hidden = mol_hidden.transpose(0, 1)  # (1, batch_size, hidden_dim)
        prot_hidden = prot_hidden.transpose(0, 1)  # (1, batch_size, hidden_dim)

        # Debug statements to check shapes
        # print(f"mol_hidden shape after transpose: {mol_hidden.shape}")
        # print(f"prot_hidden shape after transpose: {prot_hidden.shape}")

        # Cross-attention mechanism
        attention_output, _ = self.cross_attention(mol_hidden, prot_hidden, prot_hidden)
        attention_output = attention_output.transpose(0, 1).squeeze(1)  # (1, batch_size, hidden_dim) -> (batch_size, hidden_dim)

        # Debug statements to check shapes
        # print(f"attention_output shape after cross_attention: {attention_output.shape}")

        # Concatenate all representations
        t_embed = self.positional_embedding(time_step).squeeze(1)
        combined_repr = torch.cat((attention_output, t_embed), dim=1)

        # Fully connected layers with dropout
        hidden = F.relu(self.fc1(combined_repr))
        hidden = self.dropout(hidden)
        hidden = F.relu(self.fc2(hidden))
        hidden = self.dropout(hidden)
        pIC50 = self.fc3(hidden)

        return pIC50