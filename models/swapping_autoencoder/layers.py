import math
import torch
from torch import nn
from torch.nn import functional as F
from models.swapping_autoencoder.stylegan2.op.fused_act import FusedLeakyReLU
from models.swapping_autoencoder.stylegan2.model import (
    StyledConv,
    Blur,
    EqualLinear,
    EqualConv2d,
    ScaledLeakyReLU,
)


class EqualConvTranspose2d(nn.Module):
    """
    Equalized Convolutional Transpose 2D layer for StyleGAN2.
    """
    
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size**2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv_transpose2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out


class ConvLayer(nn.Sequential):
    """
    Convolutional layer with optional activation and bias adapted for StyleGAN2
    and swapping autoencoder models.
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        upsample=False,
        downsample=False,
        blur_kernel=(1, 3, 3, 1),
        bias=True,
        activate=True,
        padding="zero",
    ):
        layers = []

        self.padding = 0
        stride = 1

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2

        if upsample:
            layers.append(
                EqualConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=0,
                    stride=2,
                    bias=bias and not activate,
                )
            )

            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

        else:
            if not downsample:
                if padding == "zero":
                    self.padding = (kernel_size - 1) // 2

                elif padding == "reflect":
                    padding = (kernel_size - 1) // 2

                    if padding > 0:
                        layers.append(nn.ReflectionPad2d(padding))

                    self.padding = 0

                elif padding != "valid":
                    raise ValueError('Padding should be "zero", "reflect", or "valid"')

            layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                )
            )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class StyledResBlock(nn.Module):
    """
    Styled residual block for StyleGAN2 model.
    """
    
    def __init__(
        self, in_channel, out_channel, style_dim, upsample, blur_kernel=(1, 3, 3, 1)
    ):
        super().__init__()

        self.conv1 = StyledConv(
            in_channel,
            out_channel,
            3,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
        )

        self.conv2 = StyledConv(out_channel, out_channel, 3, style_dim)

        if upsample or in_channel != out_channel:
            self.skip = ConvLayer(
                in_channel,
                out_channel,
                1,
                upsample=upsample,
                blur_kernel=blur_kernel,
                bias=False,
                activate=False,
            )

        else:
            self.skip = None

    def forward(self, input, style, noise=None):
        out = self.conv1(input, style, noise)
        out = self.conv2(out, style, noise)

        if self.skip is not None:
            skip = self.skip(input)

        else:
            skip = input

        return (out + skip) / math.sqrt(2)


class ResBlock(nn.Module):
    """
    Residual block adapted for StyleGAN2 model.
    """
    
    def __init__(
        self,
        in_channel,
        out_channel,
        downsample=False,
        upsample=False,
        padding="zero",
        blur_kernel=(1, 3, 3, 1),
    ):
        super().__init__()
        if downsample and upsample:
            raise ValueError("Inconsistent upsample and downsample")

        self.conv1 = ConvLayer(in_channel, out_channel, 3, padding=padding)

        self.conv2 = ConvLayer(
            out_channel,
            out_channel,
            3,
            downsample=downsample,
            upsample=upsample,
            padding=padding,
            blur_kernel=blur_kernel,
        )

        if downsample or upsample or in_channel != out_channel:
            self.skip = ConvLayer(
                in_channel,
                out_channel,
                1,
                downsample=downsample,
                upsample=upsample,
                blur_kernel=blur_kernel,
                bias=False,
                activate=False,
            )

        else:
            self.skip = None

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        if self.skip is not None:
            skip = self.skip(input)

        else:
            skip = input

        # print(out.shape)

        return (out + skip) / math.sqrt(2)


class Stem(nn.Module):
    def __init__(self, channel, gray=False):
        super().__init__()
        if gray:
            self.stem = ConvLayer(1, channel, 3)
        else:
            self.stem = ConvLayer(3, channel, 3)

    def forward(self, x):
        return self.stem(x)


class StrBranch(nn.Module):
    """
    Structure branch of the pyramid encoder, working with the swapping autoencoder model
    to extract the structure representation from the input image.
    """
    def __init__(self, channel, structure_channel=8):
        super().__init__()

        scale1 = []
        in_channel = channel
        for i in range(0, 1):
            ch = channel * (2**i)
            scale1.append(ResBlock(in_channel, ch, downsample=True, padding="reflect"))
            in_channel = ch
        self.scale1 = nn.Sequential(*scale1)

        scale2 = []
        for i in range(1, 2):
            ch = channel * (2**i)
            scale2.append(ResBlock(in_channel, ch, downsample=True, padding="reflect"))
            in_channel = ch
        self.scale2 = nn.Sequential(*scale2)

        scale3 = []
        for i in range(2, 4):
            ch = channel * (2**i)
            scale3.append(ResBlock(in_channel, ch, downsample=True, padding="reflect"))
            in_channel = ch
        self.scale3 = nn.Sequential(*scale3)

        self.structure = nn.Sequential(
            ConvLayer(ch, ch, 1), ConvLayer(ch, structure_channel, 1)
        )

    def forward(self, input, multi_out):
        scale1 = self.scale1(input)
        scale2 = self.scale2(scale1)
        scale3 = self.scale3(scale2)
        structure = self.structure(scale3)

        if multi_out:
            return scale1, scale2, scale3, structure
        else:
            return structure


class TexBranch(nn.Module):
    """
    Texture branch of the pyramid encoder, working with the swapping autoencoder model
    to extract the texture representation from the input image.
    """
    
    def __init__(self, channel, texture_channel=8):
        super().__init__()

        scale1 = []
        in_channel = channel
        for i in range(0, 1):
            ch = channel * (2**i)
            scale1.append(ResBlock(in_channel, ch, downsample=True, padding="reflect"))
            in_channel = ch
        self.scale1 = nn.Sequential(*scale1)

        scale2 = []
        for i in range(1, 2):
            ch = channel * (2**i)
            scale2.append(ResBlock(in_channel, ch, downsample=True, padding="reflect"))
            in_channel = ch
        self.scale2 = nn.Sequential(*scale2)

        scale3 = []
        for i in range(2, 4):
            ch = channel * (2**i)
            scale3.append(ResBlock(in_channel, ch, downsample=True, padding="reflect"))
            in_channel = ch
        self.scale3 = nn.Sequential(*scale3)

        self.texture = nn.Sequential(
            ConvLayer(ch, ch * 2, 3, downsample=True, padding="valid"),
            ConvLayer(ch * 2, ch * 4, 3, downsample=True, padding="valid"),
            nn.AdaptiveAvgPool2d(1),
            ConvLayer(ch * 4, texture_channel, 1),
        )

    def forward(self, input, multi_out=True):
        scale1 = self.scale1(input)
        scale2 = self.scale2(scale1)
        scale3 = self.scale3(scale2)
        texture = torch.flatten(self.texture(scale3), 1)

        if multi_out:
            return scale1, scale2, scale3, texture
        else:
            return texture


class MultiProjectors(nn.Module):
    """
    Projector network for the swapping autoencoder model,
    projects structural reprensentations.
    """
    
    def __init__(self, channels, use_mlp=True, norm=True):
        super().__init__()
        self.use_mlp = use_mlp
        self.norm = norm

        self.projectors = nn.ModuleList()
        for channel in channels:
            proj = nn.Sequential(
                EqualLinear(channel, channel // 2, activation="fused_lrelu"),
                EqualLinear(channel // 2, channel),
            )
            self.projectors.append(proj)

    def forward(self, feats):
        if self.use_mlp:
            projected = []
            for i, feat in enumerate(feats):
                projected.append(self.projectors[i](feat))
        else:
            projected = feats

        if self.norm:
            normed_projected = []
            for feat in projected:
                # l2 norm after projection
                norm = feat.pow(2).sum(1, keepdim=True).pow(1.0 / 2)
                normed_projected.append(feat.div(norm + 1e-7))
            return normed_projected
        else:
            return projected
