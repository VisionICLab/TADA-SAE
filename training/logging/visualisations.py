import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from matplotlib import cm


def visualize_reconstructions(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    title:str="Reconstructions",
    with_delta:bool=False,
):
    """
    Plots the original and reconstructed images side by side in a grid

    Args:
        originals (torch.Tensor): Original images, shape (N, C, H, W)
        reconstructions (torch.Tensor): Reconstructed images, shape (N, C, H, W)
        title (str): Title of the plot
        show (bool): Whether to show the plot or not
    """
    if with_delta:
        if originals.shape[1] == 1:
            originals = originals.repeat(1, 3, 1, 1)
            reconstructions = reconstructions.repeat(1, 3, 1, 1)
        diff = torch.abs(originals - reconstructions).mean(axis=1)

        dmin = torch.amin(diff, dim=(1, 2), keepdim=True)
        dmax = torch.amax(diff, dim=(1, 2), keepdim=True)
        diff = (diff - dmin) / (dmax - dmin)
        diff = torch.from_numpy(
            np.apply_along_axis(cm.inferno, 0, diff.numpy())[:, :3, :, :]
        )
        all_images = torch.cat([originals, diff, reconstructions], dim=0)
    else:
        all_images = torch.cat([originals, reconstructions], dim=0)

    grid = make_grid(all_images, nrow=len(originals), padding=2, pad_value=1)
    grid = grid.permute(1, 2, 0)
    fig, ax = plt.subplots(num=1, clear=True)
    fig.tight_layout()
    ax.set_axis_off()
    ax.set_title(title)
    grid = grid.numpy()
    ax.imshow(grid, vmin=0, vmax=1)
    return grid, ax.figure