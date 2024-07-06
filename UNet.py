import torch
import torch.nn as nn
import torch.nn.functional as F
from SinPosEmb import getTimeMLP

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

    def forward(self, x, protein_embedding, time_embedding):
        batch_size, channels, seq_len = x.size()
        x = x.permute(2, 0, 1)  # (seq_len, batch_size, channels)

        protein_embedding_proj = self.protein_proj(protein_embedding + time_embedding).unsqueeze(0)  # (1, batch_size, channels)
        
        attn_context = torch.cat([x, protein_embedding_proj.expand(seq_len, -1, -1)], dim=0)  # (seq_len + 1, batch_size, channels)

        x, _ = self.attn(x, attn_context, attn_context)
        x = x.permute(1, 2, 0)  # (batch_size, channels, seq_len)
        return x

class UNet1D(nn.Module):
    def __init__(self, input_channels, output_channels, time_embedding_dim, protein_embedding_dim, num_diffusion_steps):
        super(UNet1D, self).__init__()
        self.time_mlp1 = getTimeMLP(time_embedding_dim, num_diffusion_steps, time_embedding_dim)
        self.time_mlp2 = getTimeMLP(time_embedding_dim, num_diffusion_steps, time_embedding_dim)
        self.time_mlp3 = getTimeMLP(time_embedding_dim, num_diffusion_steps, time_embedding_dim)
        self.time_mlp4 = getTimeMLP(time_embedding_dim, num_diffusion_steps, time_embedding_dim)
        self.time_mlp5 = getTimeMLP(time_embedding_dim, num_diffusion_steps, time_embedding_dim)
        self.time_mlp6 = getTimeMLP(time_embedding_dim, num_diffusion_steps, time_embedding_dim)
        self.time_mlp7 = getTimeMLP(time_embedding_dim, num_diffusion_steps, time_embedding_dim)
        self.time_mlp8 = getTimeMLP(time_embedding_dim, num_diffusion_steps, time_embedding_dim)

        # Downsampling path
        self.down1 = ResidualBlock(input_channels, 64, protein_embedding_dim)
        self.down2 = ResidualBlock(64, 128, protein_embedding_dim)
        self.down3 = ResidualBlock(128, 256, protein_embedding_dim)
        self.down4 = ResidualBlock(256, 512, protein_embedding_dim)

        self.downconv1 = nn.Conv1d(64, 64, kernel_size=4, stride=2, padding=1)
        self.downconv2 = nn.Conv1d(128, 128, kernel_size=4, stride=2, padding=1)
        self.downconv3 = nn.Conv1d(256, 256, kernel_size=4, stride=2, padding=1)

        # Attention block
        self.attn = SelfAttentionBlock(512, protein_embedding_dim)
        
        # Upsampling path
        self.upconv1 = nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1)
        
        self.up1 = ResidualBlock(512, 256, protein_embedding_dim)
        self.up2 = ResidualBlock(256, 128, protein_embedding_dim)
        self.up3 = ResidualBlock(128, 64, protein_embedding_dim)
        self.up4 = nn.Conv1d(64, output_channels, kernel_size=3, padding=1)  # Predict noise

    def forward(self, x, time_step, protein_embedding):
        # Embed time step
        time_embedding1 = self.time_mlp1(time_step).unsqueeze(-1)  # (batch_size, embedding_dim, 1)
        time_embedding2 = self.time_mlp2(time_step).unsqueeze(-1)
        time_embedding3 = self.time_mlp3(time_step).unsqueeze(-1)
        time_embedding4 = self.time_mlp4(time_step).unsqueeze(-1)
        time_embedding5 = self.time_mlp5(time_step).unsqueeze(-1)
        time_embedding6 = self.time_mlp6(time_step).unsqueeze(-1)
        time_embedding7 = self.time_mlp7(time_step).unsqueeze(-1)
        time_embedding8 = self.time_mlp8(time_step).unsqueeze(-1)
        
        # Downsampling
        x1 = self.down1(x, protein_embedding + time_embedding1)
        x2 = self.downconv1(x1)
        x2 = self.down2(x2, protein_embedding + time_embedding2)
        x3 = self.downconv2(x2)
        x3 = self.down3(x3, protein_embedding + time_embedding3)
        x4 = self.downconv3(x3)
        x4 = self.down4(x4, protein_embedding + time_embedding4)
        
        # Attention
        x4 = self.attn(x4, protein_embedding, time_embedding5)
        
        # Upsampling
        x = self.upconv1(x4)
        x = self.up1(x, protein_embedding + time_embedding6)
        x = self.upconv2(x)
        x = self.up2(x, protein_embedding + time_embedding7)
        x = self.upconv3(x)
        x = self.up3(x, protein_embedding + time_embedding8)
        x = self.up4(x)

        return x
