import math
import torch
from torch import nn
from .layers import ConvLayer, ResBlock, EqualLinear


class Discriminator(nn.Module):
    """
    StyleGAN2 discriminator model.
    """
    def __init__(self, size, channel_multiplier=1, gray=False):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        if gray:
            convs = [ConvLayer(1, channels[size], 1)]
        else:
            convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))
        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, downsample=True))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.final_conv = ConvLayer(in_channel, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)
        out = self.final_conv(out)

        out = out.view(out.shape[0], -1)
        out = self.final_linear(out)

        return out


class CooccurDiscriminator(nn.Module):
    """
    This new cooccur discriminator is to modify some details
    First, it applies max-pooling on the n_crop
    Second, it applied average-pooling on the spatial dimension
    """

    def __init__(self, channel, size=256, gray=False):
        super().__init__()
        if gray:
            encoder = [ConvLayer(1, channel, 1)]
        else:
            encoder = [ConvLayer(3, channel, 1)]

        if size >= 32:
            ch_multiplier = (2, 4, 8, 12, 24)
            downsample = (True, True, True, True, False)
        elif size == 16:
            ch_multiplier = (2, 4, 8, 12)
            downsample = (True, True, True, False)
        elif size == 8:
            ch_multiplier = (2, 4, 8)
            downsample = (True, True, False)
        else:
            raise ValueError(
                f"Unsupported input size {size} for Cooccurv2Discriminator"
            )

        in_ch = channel
        for ch_mul, down in zip(ch_multiplier, downsample):
            encoder.append(ResBlock(in_ch, channel * ch_mul, down))
            in_ch = channel * ch_mul

        # last conv layer
        k_size = 3 if size >= 256 else 1
        encoder.append(ConvLayer(in_ch, channel * 12, k_size, padding="valid"))

        # Average pool over spatial dimension
        encoder.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.encoder = nn.Sequential(*encoder)

        self.linear = nn.Sequential(
            EqualLinear(channel * 12 * 2, channel * 32, activation="fused_lrelu"),
            EqualLinear(channel * 32, channel * 32, activation="fused_lrelu"),
            EqualLinear(channel * 32, channel * 16, activation="fused_lrelu"),
            EqualLinear(channel * 16, 1),
        )

    def forward(self, input, n_crop, reference=None, ref_batch=None, ref_input=None):
        # [batch*n_crop, channel, h, w]
        out_input = self.encoder(input)
        _, channel, height, width = out_input.shape
        # [batch, channel, h, w]
        out_input = out_input.view(-1, n_crop, channel, height, width).max(1)[0]

        # [batch, channel, h, w]
        if ref_input is None:
            ref_input = self.encoder(reference)
            _, channel, height, width = ref_input.shape
            ref_input = ref_input.view(-1, n_crop * ref_batch, channel, height, width)
            ref_input = ref_input.max(1)[0]

        out = torch.cat((out_input, ref_input), 1)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out, ref_input
