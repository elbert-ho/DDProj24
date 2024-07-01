import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaGN(nn.Module):
    def __init__(self, num_features, embedding_dim):
        super(AdaGN, self).__init__()
        self.group_norm = nn.GroupNorm(8, num_features)
        self.linear = nn.Linear(embedding_dim, num_features * 2)

    def forward(self, h, y):
        h = self.group_norm(h)
        y = self.linear(y)
        ys, yb = y.chunk(2, dim=1)
        h = ys.unsqueeze(-1) * h + yb.unsqueeze(-1)
        return h



class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super(SinusoidalPositionEmbeddings, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, time_step):
        half_dim = self.embedding_dim // 2
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
        emb = time_step.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embedding_dim):
        super(ResidualBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            AdaGN(out_channels, embedding_dim),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            AdaGN(out_channels, embedding_dim)
        )

    def forward(self, x, embedding):
        out = self.block1(x, embedding)
        out = self.block2(out, embedding)
        return F.relu(x + out)


class SelfAttentionBlock(nn.Module):
    def __init__(self, channels, embedding_dim):
        super(SelfAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads=8)
        self.protein_proj = nn.Linear(embedding_dim, channels)

    def forward(self, x, protein_embedding):
        batch_size, channels, seq_len = x.size()
        x = x.permute(2, 0, 1)  # (seq_len, batch_size, channels)

        protein_embedding_proj = self.protein_proj(protein_embedding).unsqueeze(0)  # (1, batch_size, channels)
        
        attn_context = torch.cat([x, protein_embedding_proj.expand(seq_len, -1, -1)], dim=0)  # (seq_len + 1, batch_size, channels)

        x, _ = self.attn(x, attn_context, attn_context)
        x = x.permute(1, 2, 0)  # (batch_size, channels, seq_len)
        return x



class UNet1D(nn.Module):
    def __init__(self, input_channels, output_channels, time_embedding_dim, protein_embedding_dim):
        super(UNet1D, self).__init__()
        self.time_embedding = SinusoidalPositionEmbeddings(time_embedding_dim)
        self.protein_embedding_dim = protein_embedding_dim
        
        # Downsampling path
        self.down1 = ResidualBlock(input_channels, 64, protein_embedding_dim)
        self.down2 = ResidualBlock(64, 128, protein_embedding_dim)
        self.down3 = ResidualBlock(128, 256, protein_embedding_dim)
        self.down4 = ResidualBlock(256, 512, protein_embedding_dim)
        
        # Attention block
        self.attn = SelfAttentionBlock(512, protein_embedding_dim)
        
        # Upsampling path
        self.upconv1 = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose1d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)
        
        self.up1 = ResidualBlock(512, 256, protein_embedding_dim)
        self.up2 = ResidualBlock(256, 128, protein_embedding_dim)
        self.up3 = ResidualBlock(128, 64, protein_embedding_dim)
        self.up4 = nn.Conv1d(64, output_channels, kernel_size=3, padding=1)  # Predict noise

        self.pool = nn.MaxPool1d(2)
        self.upconv = nn.ConvTranspose1d(512, 256, kernel_size=2, stride=2)

    def forward(self, x, time_step, protein_embedding):
        # Embed time step
        time_embedding = self.time_embedding(time_step).unsqueeze(-1)  # (batch_size, embedding_dim, 1)
        
        # Downsampling
        x1 = self.down1(x, protein_embedding)
        x2 = self.pool(x1)
        x2 = self.down2(x2, protein_embedding)
        x3 = self.pool(x2)
        x3 = self.down3(x3, protein_embedding)
        x4 = self.pool(x3)
        x4 = self.down4(x4, protein_embedding)
        
        # Attention
        x4 = self.attn(x4, protein_embedding)
        
        # Upsampling
        x = self.upconv1(x4)
        x = self.up1(x, protein_embedding)
        x = self.upconv2(x)
        x = self.up2(x, protein_embedding)
        x = self.upconv3(x)
        x = self.up3(x, protein_embedding)
        x = self.up4(x)

        return x
