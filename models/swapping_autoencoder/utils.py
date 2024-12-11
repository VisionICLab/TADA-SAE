import math
import random
import torch
import cv2
import numpy as np
import torch.nn.functional as F


def get_N(W, H):
    """N that maps from unnormalized to normalized coordinates"""
    N = np.zeros((3, 3), dtype=np.float64)
    N[0, 0] = 2.0 / W
    N[0, 1] = 0
    N[1, 1] = 2.0 / H
    N[1, 0] = 0
    N[0, -1] = -1.0
    N[1, -1] = -1.0
    N[-1, -1] = 1.0
    return N


def get_N_inv(W, H):
    """N that maps from normalized to unnormalized coordinates"""
    N = get_N(W, H)
    return np.linalg.inv(N)


def cvt_MToTheta(M, w, h):
    """convert affine warp matrix `M` compatible with `opencv.warpAffine` to `theta` matrix
    compatible with `torch.F.affine_grid`
    Note:
    M works with `opencv.warpAffine`.
    To transform a set of bounding box corner points using `opencv.perspectiveTransform`, M^-1 is required
    Parameters
    ----------
    M : np.ndarray
        affine warp matrix shaped [2, 3]
    w : int
        width of image
    h : int
        height of image
    Returns
    -------
    np.ndarray
        theta tensor for `torch.F.affine_grid`, shaped [2, 3]
    """
    M_aug = np.concatenate([M, np.zeros((1, 3))], axis=0)
    M_aug[-1, -1] = 1.0
    N = get_N(w, h)
    N_inv = get_N_inv(w, h)
    theta = N @ M_aug @ N_inv
    theta = np.linalg.inv(theta)
    return theta[:2, :]


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def differential_crop_w_random_rotation(image, max_angle):
    """
    max_angle defines the maximum rotation angles from [-max_angle, max_angle]
    """
    thetas = []
    N, D, H, W = image.shape
    for i in range(N):
        angle = random.uniform(-max_angle, max_angle)
        center = (W // 2, H // 2)
        scale = (
            math.cos(math.pi * abs(angle) / 180)
            + math.sin(math.pi * abs(angle) / 180) * H / W
        )
        affine_trans = cv2.getRotationMatrix2D(center, angle, scale)
        theta = cvt_MToTheta(affine_trans, W, H)
        thetas.append(torch.from_numpy(theta))
    thetas = torch.stack(thetas).to(image.device)
    grid = F.affine_grid(thetas, image.size(), align_corners=False)
    rotated = F.grid_sample(image, grid.float(), align_corners=False)

    return rotated


def raw_patchify_image(
    img, n_crop, mask=None, min_size=1 / 16, max_size=1 / 8, max_angle=60
):
    """
    Decomposes the image in patches of random size in a given mask.

    Args:
        img (torch.Tensor): input image, shape [batch, channel, height, width]
        n_crop (int): number of patches to sample
        mask (torch.Tensor): mask to direct the sampling, shape [batch, 1, height, width]. If None, only get
        non-zero patches.
        min_size (float): minimum size of the patch, relative to the image size
        max_size (float): maximum size of the patch, relative to the image size
        max_angle (float): maximum rotation angle of the patch, in degrees

    Returns:
        torch.Tensor: sampled patches, shape [batch*n_crop, channel, target_h, target_w]
    """
    batch, channel, height, width = img.shape

    def compute_corner(cent_x, cent_y, c_w, c_h):
        c_x = cent_x - c_w // 2
        c_y = cent_y - c_h // 2
        return c_x, c_y

    default_mask = torch.zeros(batch, height, width)
    default_mask[:, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 1
    if mask is None:
        mask = default_mask
    mask = mask.squeeze(1)

    target_h = int(height * max_size)
    target_w = int(width * max_size)

    patches = []
    for b in range(batch):
        indices = torch.nonzero(mask[b])  # [height, width]
        if indices.size(0) == 0:
            indices = torch.nonzero(default_mask[b])
        for _ in range(n_crop):
            # random sample origin
            ind = random.randrange(0, indices.size(0))
            out = indices[ind].tolist()
            cent_y, cent_x = out
            # random sample crop size
            crop_ratio = random.uniform(0, 1) * (max_size - min_size) + min_size
            c_h, c_w = int(height * crop_ratio), int(width * crop_ratio)
            # recompute corners and area
            c_x, c_y = compute_corner(cent_x, cent_y, c_w, c_h)
            # clip the coordinates
            if c_y < 0:
                c_y = 0
            if c_x < 0:
                c_x = 0
            if c_y + c_h >= height:
                c_y = height - c_h - 1
            if c_x + c_w >= width:
                c_x = width - c_w - 1

            init_patch = img[b, :, c_y : c_y + c_h, c_x : c_x + c_w].view(
                1, channel, c_h, c_w
            )
            intp_patch = F.interpolate(
                init_patch,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
            patches.append(intp_patch)
    patches = torch.cat(patches, dim=0)
    rotated = differential_crop_w_random_rotation(patches, max_angle)

    return rotated, mask


def sample_patches(feat_list, n_crop, mask=None, coords=None, inv=False):
    if inv and mask is not None:
        mask = torch.logical_not(mask)
    # if mask is not None:
    #     mask = mask.squeeze()

    # collect info
    batchSize = feat_list[0].size(0)
    channels = []
    spt_dims = []
    for feat in feat_list:
        assert feat.size(0) == batchSize, "Batch size of features should be consistent"
        channels.append(feat.size(1))
        spt_dims.append(feat.shape[2:])

    if coords is None:
        # sample coords from mask
        batch_coords = []
        for b in range(batchSize):
            # height and width of mask
            h, w = mask[b].shape
            # get valid candidates from mask
            indices = torch.nonzero(mask[b])
            if indices.size(0) == 0:
                indices = torch.nonzero(torch.ones(h, w))
            # sample points
            normed_coords = []
            for _ in range(n_crop):
                ind = random.randrange(0, indices.size(0))
                cent_y, cent_x = indices[ind].tolist()
                normed_coords.append((cent_y / h, cent_x / w))
            batch_coords.append(normed_coords)
    else:
        batch_coords = coords

    # extract features according to sampled points
    scale_feats = []
    for i, feat in enumerate(feat_list):
        h, w = spt_dims[i]
        sampled_feats = []
        for b in range(batchSize):
            for j in range(n_crop):
                cent_y, cent_x = min(h - 1, int(batch_coords[b][j][0] * h)), min(
                    w - 1, int(batch_coords[b][j][1] * w)
                )
                sampled_feats.append(feat[b, :, cent_y, cent_x])

        sampled_feats = torch.stack(sampled_feats, dim=0)
        scale_feats.append(sampled_feats)

    return scale_feats, batch_coords
