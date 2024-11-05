import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class MultiTaskTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout, num_tasks):
        super(MultiTaskTransformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

        self.task_heads = nn.ModuleList([nn.Linear(d_model * 3, 1) for _ in range(num_tasks)])

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2).to(src.device)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3).to(tgt.device)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(tgt.device)
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for i, enc_layer in enumerate(self.encoder_layers):
            enc_output = enc_layer(enc_output, src_mask)
            # if i == len(self.encoder_layers) - 2:
                # penultimate_output = enc_output

        enc_output = torch.tanh(enc_output)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)

        # Aggregate token representations using concatenation strategy
        mean_pool = enc_output.mean(dim=1)  # Mean pooling
        max_pool = enc_output.max(dim=1)[0]  # Max pooling
        first_token_last_layer = enc_output[:, 0, :]  # First token output of the last layer
        # first_token_penultimate_layer = penultimate_output[:, 0, :]  # First token output of the penultimate layer

        # Concatenate the vectors
        token_representations = torch.cat([mean_pool, max_pool, first_token_last_layer], dim=1)  # Shape: (batch_size, d_model * 4)

        # Compute task outputs
        task_outputs = torch.cat([head(token_representations) for head in self.task_heads], dim=1)  # Shape: (batch_size, num_tasks)

        return output, task_outputs

    def encode_smiles(self, src):
        src = src.to(torch.long)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        enc_output = src_embedded

        first_token_penultimate_layer = None # First token output of the penultimate layer
        for i, enc_layer in enumerate(self.encoder_layers):
            enc_output = enc_layer(enc_output, (src != 0).unsqueeze(1).unsqueeze(2).to(src.device))
            # if i == len(self.encoder_layers) - 2:
                # penultimate_output = enc_output
                # first_token_penultimate_layer = penultimate_output[:, 0, :]

        enc_output = torch.tanh(enc_output)

        mean_pool = enc_output.mean(dim=1)  # Mean pooling
        max_pool = enc_output.max(dim=1)[0]  # Max pooling
        first_token_last_layer = enc_output[:, 0, :]  # First token output of the last layer
        # first_token_penultimate_layer = penultimate_output[:, 0, :]  # First token output of the penultimate layer

        # Concatenate the vectors
        fingerprint = torch.cat([mean_pool, max_pool, first_token_last_layer], dim=1)  # Shape: (batch_size, d_model * 4)

        return enc_output, fingerprint

    def decode_representation(self, enc_output, fingerprint, max_length, tokenizer):
        batch_size = enc_output.size(0)
        decoded_sequence = torch.full((batch_size, 1), tokenizer.token_to_id("[CLS]"), dtype=torch.long, device=enc_output.device)
        eos_token_id = tokenizer.token_to_id("[EOS]")

        finished = torch.zeros(batch_size, dtype=torch.bool, device=enc_output.device)

        for _ in range(max_length):
            tgt_embedded = self.positional_encoding(self.decoder_embedding(decoded_sequence))
            tgt_mask = (decoded_sequence != 0).unsqueeze(1).unsqueeze(2).to(decoded_sequence.device)
            seq_length = decoded_sequence.size(1)
            nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool().to(decoded_sequence.device)
            tgt_mask = tgt_mask & nopeak_mask

            dec_output = tgt_embedded
            for dec_layer in self.decoder_layers:
                dec_output = dec_layer(dec_output, enc_output, None, tgt_mask)

            output = self.fc(dec_output)
            _, next_token = torch.max(output[:, -1, :], dim=-1)

            decoded_sequence = torch.cat((decoded_sequence, next_token.unsqueeze(1)), dim=1)
            finished |= (next_token == eos_token_id)

            if finished.all():
                break

        for i in range(decoded_sequence.size(0)):
            eos_index = (decoded_sequence[i] == eos_token_id).nonzero(as_tuple=True)[0]
            if eos_index.numel() > 0:
                decoded_sequence[i, eos_index[0]+1:] = tokenizer.token_to_id("[PAD]")

        task_outputs = None
        if fingerprint is not None:
            # Compute task outputs
            task_outputs = torch.cat([head(fingerprint) for head in self.task_heads], dim=1)  # Shape: (batch_size, num_tasks)

        return decoded_sequence, task_outputs
