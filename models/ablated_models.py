import torch
from torch import nn

class LSAEClassifierFullIm(nn.Module):

    def __init__(self, trained_tex_encoder, texture_dims=1024, device='cuda'):
        """
        From ablation study on DMRIR (LSAE full im.)
        Trainable linear head on texture features from frozen encoder
        """

        super().__init__()
        self.trained_tex_encoder = trained_tex_encoder.eval()
        self.head = nn.Sequential(
            nn.Linear(texture_dims,1),
            nn.Sigmoid()
        )
        self.device = device
        
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Full thermal image without applied mask
        """

        with torch.no_grad():
            _, t = self.trained_tex_encoder(x.to(self.device), run_str=False, multi_tex=False)
        out = self.head(t)
        return out

    
class LSAEClassifierLeftRight(LSAEClassifierFullIm):
    def __init__(self, trained_tex_encoder, texture_dims=1024, device='cuda'):
        """
        From ablation study on DMRIR (LSAE left-right)
        Trainable linear head on left-right texture features from frozen encoder
        """
        super().__init__(trained_tex_encoder, texture_dims, device)

    def forward(self, x):
        l_x, r_x = x
        with torch.no_grad():
            _, l_t = self.trained_tex_encoder(l_x.to(self.device), run_str=False, multi_tex=False)
            _, r_t = self.trained_tex_encoder(r_x.to(self.device), run_str=False, multi_tex=False)
            feats = torch.abs(l_t - r_t)
        out = self.head(feats)
        return out