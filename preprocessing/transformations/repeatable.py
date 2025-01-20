import numpy as np
from torch import nn
import torch
import torchvision.transforms.v2.functional as TF


class RepeatableTransform(nn.Module):
    """
    A base class for freezing the randomness in
    data augmentation util .step() is called.

    Used for ADAAUG, where it is more stable to apply the same set of
    augmentations on multiple images and masks, something not yet implemented
    in torchvision.transforms.v2 (2024).   
    """

    def __init__(self, im_shape=None):
        super().__init__()
        self.im_shape = im_shape

    def step(self):
        pass


class Translation(RepeatableTransform):
    def __init__(self, translate_factor, im_shape=(256, 256)):
        super().__init__(im_shape)
        self.translate_factor = translate_factor
        self.current_translate = [0, 0]
        self.step()

    @torch.no_grad()
    def forward(self, img, mask):
        img = TF.affine(
            img,
            0,
            self.current_translate,
            1.0,
            0,
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        mask = TF.affine(
            mask,
            0,
            self.current_translate,
            1.0,
            0,
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        return img, mask

    def step(self):
        self.current_translate = np.random.uniform(
            -self.translate_factor, self.translate_factor, 2
        )
        self.current_translate *= np.array(self.im_shape)
        self.current_translate = self.current_translate.tolist()


class Rotate(RepeatableTransform):
    def __init__(self, angle):
        super().__init__()
        self.angle = angle
        self.current_angle = 0
        self.step()

    @torch.no_grad()
    def forward(self, img, mask):
        img = TF.affine(
            img,
            self.current_angle,
            [0, 0],
            1.0,
            0,
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        mask = TF.affine(
            mask,
            self.current_angle,
            [0, 0],
            1.0,
            0,
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        return img, mask

    def step(self):
        self.current_angle = np.random.uniform(-self.angle, self.angle)


class Shear(RepeatableTransform):
    def __init__(self, shear):
        super().__init__()
        self.shear = shear
        self.current_shear = [0, 0]
        self.step()

    @torch.no_grad()
    def forward(self, img, mask):
        img = TF.affine(
            img,
            0,
            [0, 0],
            1.0,
            self.current_shear,
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        mask = TF.affine(
            mask,
            0,
            [0, 0],
            1.0,
            self.current_shear,
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        return img, mask

    def step(self):
        self.current_shear = np.random.uniform(-self.shear, self.shear, 2).tolist()


class Brightness(RepeatableTransform):
    def __init__(self, brightness):
        super().__init__()
        self.brightness = brightness
        self.current_brightness = 0
        self.step()

    @torch.no_grad()
    def forward(self, img, mask):
        im_min = img.min()
        im_max = img.max()

        # Adjust the brightness for image only, mask passes through
        img += self.current_brightness
        img = torch.clamp(img, im_min, im_max)
        return img, mask

    def step(self):
        self.current_brightness = np.random.uniform(-self.brightness, self.brightness)


class ResizeCrop(RepeatableTransform):
    def __init__(self, factor=0.5, im_shape=(256, 256)):
        super().__init__(im_shape)
        self.factor = factor
        self.top = None
        self.left = None
        self.width = None
        self.height = None
        self.step()

    @torch.no_grad()
    def forward(self, img, mask):
        img = TF.resized_crop(
            img, self.top, self.left, self.width, self.height, self.im_shape
        )
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
        mask = TF.resized_crop(
            mask, self.top, self.left, self.width, self.height, self.im_shape
        )
        return img, mask

    def step(self):
        h, w = self.im_shape
        self.top = np.random.randint(0, int(h * self.factor))
        self.left = np.random.randint(0, int(w * self.factor))
        self.height = np.random.randint(int(h * self.factor), h)
        self.width = np.random.randint(int(w * self.factor), w)


class Perspective(RepeatableTransform):
    def __init__(self, distortion_scale, im_shape=(256, 256)):
        super().__init__(im_shape)
        self.distortion_scale = distortion_scale
        self.h, self.w = self.im_shape
        self.startpoints = None
        self.endpoints = None
        self.step()

    @torch.no_grad()
    def forward(self, img, mask):
        img = TF.perspective(img, self.startpoints, self.endpoints)
        mask = TF.perspective(mask, self.startpoints, self.endpoints)
        return img, mask

    def step(self):
        half_height = self.h // 2
        half_width = self.w // 2
        bound_height = int(self.distortion_scale * half_height) + 1
        bound_width = int(self.distortion_scale * half_width) + 1
        topleft = [
            int(torch.randint(0, bound_width, size=(1,))),
            int(torch.randint(0, bound_height, size=(1,))),
        ]
        topright = [
            int(torch.randint(self.w - bound_width, self.w, size=(1,))),
            int(torch.randint(0, bound_height, size=(1,))),
        ]
        botright = [
            int(torch.randint(self.h - bound_width, self.w, size=(1,))),
            int(torch.randint(self.h - bound_height, self.h, size=(1,))),
        ]
        botleft = [
            int(torch.randint(0, bound_width, size=(1,))),
            int(torch.randint(self.h - bound_height, self.h, size=(1,))),
        ]
        self.startpoints = [
            [0, 0],
            [self.w - 1, 0],
            [self.w - 1, self.h - 1],
            [0, self.h - 1],
        ]
        self.endpoints = [topleft, topright, botright, botleft]


class Elastic(RepeatableTransform):
    def __init__(self, alpha, sigma, im_shape=(256, 256)):
        super().__init__(im_shape)
        self.alpha = alpha
        self.sigma = sigma
        self.displacement = None
        self.step()

    @torch.no_grad()
    def forward(self, img, mask):
        img = TF.elastic_transform(img, self.displacement)
        mask = TF.elastic_transform(mask, self.displacement)
        return img, mask

    def step(self):
        size = list(self.im_shape)
        dx = torch.rand([1, 1] + size) * 2 - 1
        if self.sigma > 0.0:
            kx = int(8 * self.sigma + 1)
            # if kernel size is even we have to make it odd
            if kx % 2 == 0:
                kx += 1
            dx = TF.gaussian_blur(dx, [kx, kx], [self.sigma])
        dx = dx * self.alpha / size[0]

        dy = torch.rand([1, 1] + size) * 2 - 1
        if self.sigma > 0.0:
            ky = int(8 * self.sigma + 1)
            # if kernel size is even we have to make it odd
            if ky % 2 == 0:
                ky += 1
            dy = TF.gaussian_blur(dy, [ky, ky], [self.sigma])
        dy = dy * self.alpha / size[1]
        self.displacement = torch.concat([dx, dy], 1).permute(
            [0, 2, 3, 1]
        )  # 1 x H x W x 2


class GaussianNoise(RepeatableTransform):
    def __init__(self, std_range=(0, 3), im_shape=(256, 256)):
        super().__init__(im_shape)
        self.std_range = std_range
        self.noise = None
        self.current_std = None
        self.step()

    @torch.no_grad()
    def forward(self, img, mask):
        img = img + self.noise.to(img.device)
        return img, mask

    def step(self):
        self.current_std = np.random.uniform(*self.std_range)
        self.noise = torch.randn(self.im_shape) * self.current_std


class Gamma(RepeatableTransform):
    def __init__(self, gamma_range=(0.5, 2)):
        super().__init__(None)
        self.gamma_range = gamma_range
        self.current_gamma = None
        self.step()

    @torch.no_grad()
    def forward(self, img, mask):
        img = TF.adjust_gamma(img, self.current_gamma)
        return img, mask

    def step(self):
        self.current_gamma = np.random.uniform(*self.gamma_range)


class Sharpen(RepeatableTransform):
    def __init__(self, sharpen_factor_range=(0, 1)):
        super().__init__(None)
        self.sharpen_factor_range = sharpen_factor_range
        self.current_sharpen_factor = None
        self.step()

    @torch.no_grad()
    def forward(self, img, mask):
        img = TF.adjust_sharpness(img, self.current_sharpen_factor)
        return img, mask

    def step(self):
        self.current_sharpen_factor = np.random.uniform(*self.sharpen_factor_range)

