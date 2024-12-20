import torch
from torch import nn

class LSAEClassifierFullIm(nn.Module):
    def __init__(self, encoder, texture_dims, device='cuda'):
        super().__init__()
        self.encoder = encoder.eval()
        self.head = nn.Sequential(
            nn.Linear(texture_dims,1),
            nn.Sigmoid()
        )
        self.device = device
    
    def forward(self, x):
        with torch.no_grad():
            _, t = self.encoder(x.to(self.device), run_str=False, multi_tex=False)
        out = self.head(t)
        return out
    
class LSAEClassifierLeftRight(nn.Module):
    def __init__(self, encoder, texture_dims, device='cuda'):
        """
        Linear probing class for the LSAE model
        """
        super().__init__()
        self.encoder = encoder.eval()
        self.head = nn.Sequential(
            nn.Linear(texture_dims,1),
            nn.Sigmoid()
        )
        self.device = device

    def forward(self, x):
        l_x, r_x = x
        with torch.no_grad():
            _, l_t = self.encoder(l_x.to(self.device), run_str=False, multi_tex=False)
            _, r_t = self.encoder(r_x.to(self.device), run_str=False, multi_tex=False)
            feats = torch.abs(l_t - r_t)
        out = self.head(feats)
        return out