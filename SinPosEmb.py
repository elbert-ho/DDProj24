import math
import torch
import torch.nn as nn

class SinPosEmb(nn.Module):
    def __init__(self, dim, theta):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def getTimeMLP(dim, theta, time_dim):
    dim = int(dim)  # Ensure dim is an integer
    model = nn.Sequential(
            SinPosEmb(dim, theta),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
            )
    model = model.to('cuda')
    return model