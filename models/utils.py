import torch
from torch import nn
import numpy as np
import locale
from tqdm import trange


def count_parameters(model):
    """
    Returns the number of trainable parameters in a model.
    From https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
    """
    locale.setlocale(locale.LC_ALL, "en_GB.UTF-8")
    num_params = sum(p.numel() for p in model.parameters())
    return locale.format_string("%d", num_params, True)


def get_dataset_latents(encoder, dataset, device="cpu", progress=False):
    all_feats = []
    iterator = trange if progress else range
    for j in iterator(len(dataset)):
        im = dataset[j].unsqueeze(0).to(device)
        feats = encoder(im).detach().flatten(1).cpu().numpy()
        all_feats.append(feats)
    return np.concatenate(all_feats)


class GradientPenalty(nn.Module):
    """
    From https://github.com/EmilienDupont/wgan-gp/blob/master/training.py

    """

    def __init__(self, discriminator, gp_weight=1):
        super(GradientPenalty, self).__init__()
        self.discriminator = discriminator
        self.gp_weight = gp_weight

    def forward(self, gen, real):
        bs = real.size(0)
        alpha = torch.rand(bs, 1, 1, 1).to(real.device)
        alpha = alpha.expand_as(real)
        interpolated = alpha * real.data + (1 - alpha) * gen.data
        interpolated = torch.autograd.Variable(interpolated, requires_grad=True).to(
            real.device
        )

        # Calculate probability of interpolated examples
        prob_interpolated = self.discriminator(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(prob_interpolated).to(real.device),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradients = gradients.view(bs, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
        return self.gp_weight * ((gradients_norm - 1) ** 2).mean()


def cal_gradient_penalty(
    netD, real_data, fake_data, device, type="mixed", constant=1.0, lambda_gp=10.0
):
    """
    From: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py

    Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if (
            type == "real"
        ):  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == "fake":
            interpolatesv = fake_data
        elif type == "mixed":
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = (
                alpha.expand(
                    real_data.shape[0], real_data.nelement() // real_data.shape[0]
                )
                .contiguous()
                .view(*real_data.shape)
            )
            interpolatesv = alpha * real_data.data + ((1 - alpha) * fake_data.data)
        else:
            raise NotImplementedError("{} not implemented".format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolatesv,
            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (
            ((gradients + 1e-16).norm(2, dim=1) - constant) ** 2
        ).mean() * lambda_gp  # added eps
        return gradient_penalty
    else:
        return 0.0
