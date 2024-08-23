import torch
import torch.nn as nn
import torch.nn.functional as F
from SinPosEmb import getTimeMLP

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.LayerNorm(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm2 = nn.LayerNorm(out_channels)
        
        # If in_channels != out_channels, adjust dimensions for the skip connection
        self.skip_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        
        if self.skip_conv is not None:
            residual = self.skip_conv(residual)
            
        out += residual  # Add the input (residual) to the output
        return F.relu(out)  # Apply ReLU after the addition

class pIC50Predictor(nn.Module):
    def __init__(self, max_timesteps, protein_dim):
        super(pIC50Predictor, self).__init__()
        
        self.block1 = ResidualBlock(in_channels=256, out_channels=128)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=0.2)

        self.block2 = ResidualBlock(in_channels=128, out_channels=64)
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=0.2)

        self.block3 = ResidualBlock(in_channels=64, out_channels=32)
        self.pool3 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(p=0.2)
        
        self.mol_projection = nn.Linear(512, 512)

        self.time_embedding = getTimeMLP(128, max_timesteps, 512)
        self.prot_projection = nn.Linear(protein_dim, 512)

        self.fc1 = nn.Linear(512 * 3, 1024)
        self.norm4 = nn.LayerNorm(1024)
        self.dropout4 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, 1024)
        self.norm5 = nn.LayerNorm(1024)
        self.dropout5 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(1024, 1024)
        self.norm6 = nn.LayerNorm(1024)
        self.dropout6 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(1024, 1)

    def forward(self, x, prot, t):
        # Input size: x = [B, 256, 128], prot = [B, 1280], t = [B,]
        x = self.block1(x)  # After block1: [B, 128, 128]
        x = self.pool1(x)   # After pool1: [B, 128, 64]
        x = self.dropout1(x)  # After dropout1: [B, 128, 64]

        x = self.block2(x)  # After block2: [B, 64, 64]
        x = self.pool2(x)   # After pool2: [B, 64, 32]
        x = self.dropout2(x)  # After dropout2: [B, 64, 32]

        x = self.block3(x)  # After block3: [B, 32, 32]
        x = self.pool3(x)   # After pool3: [B, 32, 16]
        x = self.dropout3(x)  # After dropout3: [B, 32, 16]

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)  # After flattening: [B, 32 * 16]
        x = self.mol_projection(x)
        
        prot = self.prot_projection(prot)
        time_proj = self.time_embedding(t)
        
        combined = torch.cat((x, prot, time_proj), dim=1)

        combined = self.dropout4(F.relu(self.norm4(self.fc1(combined))))
        combined = self.dropout5(F.relu(self.norm5(self.fc2(combined))))
        combined = self.dropout6(F.relu(self.norm6(self.fc3(combined))))
        combined = self.fc4(combined)
        return combined
