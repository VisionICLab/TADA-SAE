import torch
from torch import nn

class LinearClassificationHead(nn.Module):
    def __init__(self, texture_dims=1024):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(texture_dims,1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.head(x)


class ConvEncoder(nn.Module):
    """
    A convolutional encoder compressing the input from the original input space to a latent space.
    The compression ratio is equal to c_hid/z_dim.
    From: https://github.com/plutoyuxie/AutoEncoder-SSIM-for-unsupervised-anomaly-detection-/blob/master/network.py

    Args:
        z_dim (int): The dimension of the latent space.
        c_hid (int): The number of hidden channels.
        input_size (tuple): The size of the input image (channels, width, height).
        act_fn (torch.nn.Module): The activation function to use between each layer..
    """

    def __init__(self, z_dim, c_hid, c_in, act_fn=nn.LeakyReLU(0.2)):
        super(ConvEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(c_in, c_hid, kernel_size=4, stride=2, padding=1),
            act_fn,
            nn.Conv2d(c_hid, c_hid, kernel_size=4, stride=2, padding=1),
            act_fn,
            nn.Conv2d(c_hid, c_hid, kernel_size=3, stride=1, padding=1),
            act_fn,
            nn.Conv2d(c_hid, c_hid * 2, kernel_size=4, stride=2, padding=1),
            act_fn,
            nn.Conv2d(c_hid * 2, c_hid * 2, kernel_size=3, stride=1, padding=1),
            act_fn,
            nn.Conv2d(c_hid * 2, c_hid * 4, kernel_size=4, stride=2, padding=1),
            act_fn,
            nn.Conv2d(c_hid * 4, c_hid * 2, kernel_size=3, stride=1, padding=1),
            act_fn,
            nn.Conv2d(c_hid * 2, c_hid, kernel_size=3, stride=1, padding=1),
            act_fn,
            nn.Conv2d(c_hid, z_dim, kernel_size=8, stride=1, padding=0),
        )

    def forward(self, x):
        return self.encoder(x)

class ConvDecoder(nn.Module):
    """
    A convolutional decoder decompressing the input from a latent space to the original input space.
    The compression ratio is equal to c_hid/z_dim.
    From: https://github.com/plutoyuxie/AutoEncoder-SSIM-for-unsupervised-anomaly-detection-/blob/master/network.py

    Args:
        z_dim (int): The dimension of the latent space.
        c_hid (int): The number of hidden channels.
        input_size (tuple): The size of the input image (channels, width, height).
        act_fn (torch.nn.Module): The activation function to use between each layer.
        output_act_fn (torch.nn.Module): The activation function to use at the output layer.
    """

    def __init__(
        self, z_dim, c_hid, c_in, act_fn=nn.LeakyReLU(0.1), output_act_fn=nn.Sigmoid()
    ):
        super(ConvDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, c_hid, kernel_size=8, stride=1, padding=0),
            act_fn,
            nn.Conv2d(c_hid, c_hid * 2, kernel_size=3, stride=1, padding=1),
            act_fn,
            nn.Conv2d(c_hid * 2, c_hid * 4, kernel_size=3, stride=1, padding=1),
            act_fn,
            nn.ConvTranspose2d(
                c_hid * 4, c_hid * 2, kernel_size=4, stride=2, padding=1
            ),
            act_fn,
            nn.Conv2d(c_hid * 2, c_hid * 2, kernel_size=3, stride=1, padding=1),
            act_fn,
            nn.ConvTranspose2d(c_hid * 2, c_hid, kernel_size=4, stride=2, padding=1),
            act_fn,
            nn.Conv2d(c_hid, c_hid, kernel_size=3, stride=1, padding=1),
            act_fn,
            nn.ConvTranspose2d(c_hid, c_hid, kernel_size=4, stride=2, padding=1),
            act_fn,
            nn.ConvTranspose2d(c_hid, c_in, kernel_size=4, stride=2, padding=1),
            output_act_fn,
        )

    def forward(self, x):
        return self.decoder(x)

class ConvAE(nn.Module):
    """
    A convolutional autoencoder encoder compressing the input from the original input space to a latent space
    and then reconstructing the input from the latent space.
    The compression ratio is equal to c_hid/z_dim.
    From: https://github.com/plutoyuxie/AutoEncoder-SSIM-for-unsupervised-anomaly-detection-/blob/master/network.py

    Args:
        z_dim (int): The dimension of the latent space.
        c_hid (int): The number of hidden channels.
        c_in (int): The number of input channels.
        act_fn (torch.nn.Module): The activation function to use between each layer.
        output_act_fn (torch.nn.Module): The activation function to use at the output layer.
    """

    def __init__(
        self, z_dim, c_hid, c_in, act_fn=nn.LeakyReLU(0.1), output_act_fn=nn.Sigmoid()
    ):
        super(ConvAE, self).__init__()
        self.encoder = ConvEncoder(z_dim, c_hid, c_in, act_fn)
        self.decoder = ConvDecoder(z_dim, c_hid, c_in, act_fn, output_act_fn)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

        

class LSAEClassifierFullIm(nn.Module):

    def __init__(self, trained_tex_encoder, texture_dims=1024, with_linear_head=True, device='cuda'):
        """
        From ablation study on DMRIR (LSAE full im.)
        Trainable linear head on texture features from frozen encoder
        """

        super().__init__()
        self.trained_tex_encoder = trained_tex_encoder.eval()
        self.with_linear_head = with_linear_head
        self.head = LinearClassificationHead(texture_dims)
        self.device = device
        
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Full thermal image without applied mask
        """
        with torch.no_grad():
            _, out = self.trained_tex_encoder(x.to(self.device), run_str=False, multi_tex=False)
        if self.with_linear_head:
            out = self.head(out)
        return out

    
class LSAEClassifierLeftRight(LSAEClassifierFullIm):
    def __init__(self, trained_tex_encoder, texture_dims=1024, with_linear_head=True, device='cuda'):
        """
        From ablation study on DMRIR (LSAE left-right)
        Trainable linear head on left-right texture features from frozen encoder
        """
        super().__init__(trained_tex_encoder, texture_dims, with_linear_head, device)

    def forward(self, x):
        l_x, r_x = x
        with torch.no_grad():
            _, l_t = self.trained_tex_encoder(l_x.to(self.device), run_str=False, multi_tex=False)
            _, r_t = self.trained_tex_encoder(r_x.to(self.device), run_str=False, multi_tex=False)
            out = torch.abs(l_t - r_t)
            
        if self.with_linear_head:
            out = self.head(out)
        return out
