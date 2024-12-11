import torch
from torch import nn
from torchvision import tv_tensors
from torchvision.transforms import v2
import numpy as np
import preprocessing.transformations.repeatable as TR
from datasets.utils import denormalize_image


class ADAAugment(nn.Module):
    def __init__(
        self,
        aug_p=0.0,
        ada_target=0.5,
        ada_step=1e-4,
        max_proba=0.8,
        size=(256, 256),
        mean=[0.0],
        std=[1.0],
    ):
        """
        An adaptive data augmentation module. It applies a set of transformations with a probability
        that is adjusted based on the discriminator's accuracy.
        From:
        @misc{karras2020traininggenerativeadversarialnetworks,
        title={Training Generative Adversarial Networks with Limited Data}, 
        author={Tero Karras and Miika Aittala and Janne Hellsten and Samuli Laine and Jaakko Lehtinen and Timo Aila},
        year={2020},
        eprint={2006.06676},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2006.06676}, 
        }
        """
        super().__init__()
        self.aug_p = aug_p
        self.ada_step = ada_step
        self.ada_target = ada_target
        self.max_proba = max_proba
        self.size = size
        self.mean = np.array(mean)
        self.std = np.array(std)
        self._normalize = v2.Normalize(self.mean, self.std)
        self.transforms = nn.ModuleList(
            [
                TR.ResizeCrop(0.2, size),
                TR.Translation(0.3, size),
                TR.Rotate(45),
                TR.Shear(30),
                TR.Elastic(alpha=30, sigma=3, im_shape=size),
                TR.Perspective(0.5, size),
                TR.Sharpen((0.2, 2)),
                TR.Brightness(0.25),
                TR.Gamma((0.5, 2)),
                TR.GaussianNoise((0,0.1), im_shape=size)
            ]
        )

        self.current_transforms = []

    def _denorm(self, x):
        """
        Denormalize the image tensor.
        
        Args:
            x (torch.Tensor): The input image tensor.
            
        Returns:
            torch.Tensor: The denormalized image tensor.
        """
        return denormalize_image(x, self.mean, self.std)

    def step(self, disc_pred):
        """
        Single step of the adaptive augmentation.
        
        Args:
            disc_pred (torch.Tensor): The discriminator's prediction.
            
        Returns:
            None
        """
        current_accuracy = 0.5 * (1 + torch.sign(disc_pred))
        current_accuracy = current_accuracy.mean().item()
        acc_error = current_accuracy - self.ada_target
        self.aug_p = np.clip(self.aug_p + acc_error * self.ada_step, 0, self.max_proba)

        self.current_transforms = []
        for t in self.transforms:
            if np.random.rand() < self.aug_p:
                t.step()
                self.current_transforms.append(t)

    @torch.no_grad()
    def forward(self, images, masks=None):
        """
        Apply the adaptive augmentation to the input images with the corresponding masks and
        the current set of transformations given the current probability.
        
        Args:
            images (torch.Tensor): The input images.
            masks (torch.Tensor): The corresponding masks.
            
        Returns:
            torch.Tensor: The augmented images.
            torch.Tensor: The augmented masks.
        """
        
        if self.aug_p <= 0:
            return images, masks

        aug_imgs = []
        aug_masks = []

        for b in range(images.shape[0]):
            im_aug = tv_tensors.Image(images[b]).to(images.device)
            mask_aug = tv_tensors.Mask(torch.zeros_like(im_aug)).to(images.device)
            if masks is not None:
                mask_aug = tv_tensors.Mask(masks[b])

            im_aug = self._denorm(im_aug)

            for t in self.current_transforms:
                im_aug, mask_aug = t(im_aug, mask_aug)

            im_aug = self._normalize(im_aug)
            aug_imgs.append(im_aug)
            aug_masks.append(mask_aug)
        images = torch.stack(aug_imgs)
        masks = torch.stack(aug_masks)
        return images, masks
