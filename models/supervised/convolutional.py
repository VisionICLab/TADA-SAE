import torch
from torch import nn
from torchvision import models


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
        
        1. **First Conv Layer**: 
        - Input: (1, 1, 256, 256)
        - Kernel: 4, Stride: 2, Padding: 1
        - Output: $$\left\lfloor \frac{256 - 4 + 2 \times 1}{2} + 1 \right\rfloor = 128$$
        - Shape: (1, c_hid, 128, 128)

        2. **Second Conv Layer**: 
        - Input: (1, c_hid, 128, 128)
        - Kernel: 4, Stride: 2, Padding: 1
        - Output: $$\left\lfloor \frac{128 - 4 + 2 \times 1}{2} + 1 \right\rfloor = 64$$
        - Shape: (1, c_hid, 64, 64)

        3. **Third Conv Layer**: 
        - Input: (1, c_hid, 64, 64)
        - Kernel: 3, Stride: 1, Padding: 1
        - Output: $$\left\lfloor \frac{64 - 3 + 2 \times 1}{1} + 1 \right\rfloor = 64$$
        - Shape: (1, c_hid, 64, 64)

        4. **Fourth Conv Layer**: 
        - Input: (1, c_hid, 64, 64)
        - Kernel: 4, Stride: 2, Padding: 1
        - Output: $$\left\lfloor \frac{64 - 4 + 2 \times 1}{2} + 1 \right\rfloor = 32$$
        - Shape: (1, c_hid * 2, 32, 32)

        5. **Fifth Conv Layer**: 
        - Input: (1, c_hid * 2, 32, 32)
        - Kernel: 3, Stride: 1, Padding: 1
        - Output: $$\left\lfloor \frac{32 - 3 + 2 \times 1}{1} + 1 \right\rfloor = 32$$
        - Shape: (1, c_hid * 2, 32, 32)

        6. **Sixth Conv Layer**: 
        - Input: (1, c_hid * 2, 32, 32)
        - Kernel: 4, Stride: 2, Padding: 1
        - Output: $$\left\lfloor \frac{32 - 4 + 2 \times 1}{2} + 1 \right\rfloor = 16$$
        - Shape: (1, c_hid * 4, 16, 16)

        7. **Seventh Conv Layer**: 
        - Input: (1, c_hid * 4, 16, 16)
        - Kernel: 3, Stride: 1, Padding: 1
        - Output: $$\left\lfloor \frac{16 - 3 + 2 \times 1}{1} + 1 \right\rfloor = 16$$
        - Shape: (1, c_hid * 2, 16, 16)

        8. **Eighth Conv Layer**: 
        - Input: (1, c_hid * 2, 16, 16)
        - Kernel: 3, Stride: 1, Padding: 1
        - Output: $$\left\lfloor \frac{16 - 3 + 2 \times 1}{1} + 1 \right\rfloor = 16$$
        - Shape: (1, c_hid, 16, 16)

        9. **Ninth Conv Layer**: 
        - Input: (1, c_hid, 16, 16)
        - Kernel: 8, Stride: 1, Padding: 0
        - Output: $$\left\lfloor \frac{16 - 8 + 2 \times 0}{1} + 1 \right\rfloor = 9$$
        - Shape: (1, z_dim, 9, 9)

    """

    def __init__(self, z_dim, c_hid, c_in, act_fn=nn.LeakyReLU(0.2)):
        super(ConvEncoder, self).__init__()
        self.z_dim = z_dim
        self.c_hid = c_hid
        self.c_in = c_in
        self.act_fn = act_fn
        
        self.encoder = nn.Sequential(
            nn.Conv2d(self.c_in, self.c_hid, kernel_size=4, stride=2, padding=1),
            self.act_fn,
            nn.Conv2d(self.c_hid, self.c_hid, kernel_size=4, stride=2, padding=1),
            self.act_fn,
            nn.Conv2d(self.c_hid, self.c_hid, kernel_size=3, stride=1, padding=1),
            self.act_fn,
            nn.Conv2d(self.c_hid, self.c_hid * 2, kernel_size=4, stride=2, padding=1),
            self.act_fn,
            nn.Conv2d(self.c_hid * 2, self.c_hid * 2, kernel_size=3, stride=1, padding=1),
            self.act_fn,
            nn.Conv2d(self.c_hid * 2, self.c_hid * 4, kernel_size=4, stride=2, padding=1),
            self.act_fn,
            nn.Conv2d(self.c_hid * 4, self.c_hid * 2, kernel_size=3, stride=1, padding=1),
            self.act_fn,
            nn.Conv2d(self.c_hid * 2, self.c_hid, kernel_size=3, stride=1, padding=1),
            self.act_fn,
            nn.Conv2d(self.c_hid, self.z_dim, kernel_size=8, stride=1, padding=0),
        )

    def forward(self, x):
        return self.encoder(x)


class ConvSmall(nn.Module):
    def __init__(self, z_dim=128, c_hid=32, c_in=1) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            ConvEncoder(z_dim, c_hid, c_in),
            nn.Flatten(),
            nn.Linear(z_dim*(9**2), 1),  # see ConvEncoder documentation above
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)      
    

class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        resnet18_model = models.resnet18()
        resnet18_model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        resnet18_model.fc = torch.nn.Linear(512, 1)
        self.classifier = nn.Sequential(
            resnet18_model,
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.classifier(x)
    

class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.inceptionv3_model = models.inception_v3()
        self.inceptionv3_model.transform_input=False
        self.inceptionv3_model.Conv2d_1a_3x3.conv=nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=2, bias=False)
        nn.init.kaiming_normal_(self.inceptionv3_model.Conv2d_1a_3x3.conv.weight, mode='fan_out', nonlinearity='relu')
        self.inceptionv3_model.fc = nn.Linear(self.inceptionv3_model.fc.in_features, 1)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out = self.inceptionv3_model(x)[0]
        return self.sigmoid(out)
        