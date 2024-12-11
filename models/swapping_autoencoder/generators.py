import torch
from torch import nn
from .layers import StyledResBlock, ConvLayer


class Generator(nn.Module):
    """
    Generator for the StyleGAN2 model, used for swapping autoencoder.
    
    Args:
        channel (int): number of channels in the input tensor
        structure_channel (int): number of channels in the structure tensor
        texture_channel (int): number of channels in the texture tensor
        blur_kernel (tuple): kernel for the blur operation
        gray (bool): whether the image is grayscale (1 vs 3 channels)
    """
    def __init__(
        self,
        channel,
        structure_channel=8,
        texture_channel=2048,
        blur_kernel=(1, 3, 3, 1),
        gray=False,
    ):
        super().__init__()
        self.gray = gray

        ch_multiplier = (4, 8, 12, 16, 16, 16, 8, 4)
        upsample = (False, False, False, False, True, True, True, True)

        self.layers = nn.ModuleList()
        in_ch = structure_channel
        for ch_mul, up in zip(ch_multiplier, upsample):
            self.layers.append(
                StyledResBlock(
                    in_ch, channel * ch_mul, texture_channel, up, blur_kernel
                )
            )
            in_ch = channel * ch_mul

        if gray:
            self.to_img = ConvLayer(in_ch, 1, 1, activate=False)
        else:
            self.to_rgb = ConvLayer(in_ch, 3, 1, activate=False)

    def forward(self, structure, texture, noises=None):
        if noises is None:
            noises = [None] * len(self.layers)

        out = structure
        for layer, noise in zip(self.layers, noises):
            out = layer(out, texture, noise)

        if self.gray:
            out = self.to_img(out)
        else:
            out = self.to_rgb(out)

        return out
