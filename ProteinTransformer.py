import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim  # Corrected import
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, device=x.device, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)
        embedding = self.tok_embed(x) + self.pos_embed(pos)
        return self.norm(embedding)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.fc = nn.Linear(n_heads * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)

        # Assuming attn_mask is initially of shape [batch_size, seq_length]
        # Expand it to [batch_size, 1, 1, seq_length] to cover the multi-head dimension and the second sequence dimension
        # Step 1: Unsqueezing to add the necessary dimensions
        attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)  # Adds dimensions for heads and duplicate seq_length

        # Step 2: Expanding to match the shape of scores
        # We need it to be [batch_size, n_heads, seq_length, seq_length]
        # Use expand with -1 to signify not changing that dimension, and other dimensions to match scores
        attn_mask = attn_mask.expand(-1, self.n_heads, -1, -1)  # -1 in seq_length places will keep the original seq_length from attn_mask

        # Now apply the expanded mask to the scores
        scores = torch.matmul(q_s, k_s.transpose(-2, -1)) / (self.d_k ** 0.5)
        scores = scores.masked_fill(attn_mask == 0, float('-inf'))  # Apply masking

        attn = F.softmax(scores, dim=-1)
        context = torch.matmul(attn, v_s).transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)

        output = self.fc(context)
        return self.norm(output + Q), attn

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff, dropout_rate=.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.pos_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.pos_ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class BERT(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, n_heads, d_ff, n_layers):
        super(BERT, self).__init__()
        self.embedding = Embedding(vocab_size, d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_model // n_heads, d_model // n_heads, n_heads, d_ff) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        for layer in self.layers:
            embedded = layer(embedded, attention_mask)
        logits = self.fc(embedded)
        return logits

