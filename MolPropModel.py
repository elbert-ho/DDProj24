import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from SinPosEmb import getTimeMLP

class PropertyModel(nn.Module):
    def __init__(self, input_size, max_timesteps, embedding_dim):
        super(PropertyModel, self).__init__()
        self.positional_embedding = getTimeMLP(embedding_dim / 4, max_timesteps, embedding_dim)
        self.fc1 = nn.Linear(input_size + embedding_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(128, 1)

    def forward(self, x, t):
        t_embed = self.positional_embedding(t).squeeze(1)
        x = torch.cat((x, t_embed), dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.output(x)
        return x