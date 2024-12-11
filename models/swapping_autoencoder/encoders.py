from torch import nn
from .layers import Stem, StrBranch, TexBranch


class PyramidEncoder(nn.Module):
    """
    A pyramid encoder for the swapping autoencoder model.
    Outputs the structure and texture tensors as disentangled representations of the input image.
    """
    def __init__(self, channel, structure_channel=8, texture_channel=2048, gray=False):
        super().__init__()
        self.stem = Stem(channel, gray=gray)
        self.str_branch = StrBranch(channel, structure_channel)
        self.tex_branch = TexBranch(channel, texture_channel)

    def forward(
        self, input, run_str=True, run_tex=True, multi_str=False, multi_tex=True
    ):
        structures = None
        textures = None
        out = self.stem(input)
        if run_str:
            structures = self.str_branch(out, multi_out=multi_str)
        if run_tex:
            textures = self.tex_branch(out, multi_out=multi_tex)
        return structures, textures
