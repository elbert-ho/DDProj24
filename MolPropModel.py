import torch
import torch.nn as nn
import torch.nn.functional as F
from SinPosEmb import getTimeMLP

class PropertyModel(nn.Module):
    def __init__(self, input_size, max_timesteps, embedding_dim):
        super(PropertyModel, self).__init__()
        self.positional_embedding = getTimeMLP(embedding_dim / 4, max_timesteps, embedding_dim)
        
        self.fc1 = nn.Linear(input_size + embedding_dim, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.dropout1 = nn.Dropout(0.1)
        
        self.fc2 = nn.Linear(4096, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(0.1)
        
        self.fc3 = nn.Linear(1024, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.1)
        
        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.1)
        
        self.output = nn.Linear(128, 1)

    def forward(self, x, t):
        device = x.device
        self.to(device)

        x = x.squeeze(1)
        t_embed = self.positional_embedding(t).squeeze(1).to(device)
        x = torch.cat((x, t_embed), dim=1)
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout4(x)
        
        x = self.output(x)
        return x
