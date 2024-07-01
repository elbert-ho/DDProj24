import torch
import torch.nn as nn
import torch.nn.functional as F
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

        attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
        attn_mask = attn_mask.expand(-1, self.n_heads, -1, -1)

        scores = torch.matmul(q_s, k_s.transpose(-2, -1)) / (self.d_k ** 0.5)
        scores = scores.masked_fill(attn_mask == 0, float('-inf'))

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
        all_hidden_states = []
        for layer in self.layers:
            embedded = layer(embedded, attention_mask)
            all_hidden_states.append(embedded)
        logits = self.fc(embedded)
        return logits, all_hidden_states

    def encode_protein(self, protein_tokens, attention_mask):
        _, all_hidden_states = self.forward(protein_tokens, attention_mask)
        final_layer = all_hidden_states[-1]
        penultimate_layer = all_hidden_states[-2]

        cls_final = final_layer[:, 0, :]  # CLS token representation from the final layer
        cls_penultimate = penultimate_layer[:, 0, :]  # CLS token representation from the penultimate layer

        max_pool_final = torch.max(final_layer, dim=1)[0]
        mean_pool_final = torch.mean(final_layer, dim=1)

        fingerprint = torch.cat([max_pool_final, mean_pool_final, cls_final, cls_penultimate], dim=1)
        return fingerprint
