import torch
import torch.nn as nn
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure,
    LearnedPerceptualImagePatchSimilarity,
)
from datasets.utils import denormalize_image
from torchmetrics import Metric


class ReconstructionMetric(nn.Module):
    """
    A base class for reconstruction metrics. It takes a metric and normalizes the input images
    
    Args:
        metric (Metric): a metric to calculate
        mean (list): mean of the dataset
        std (list): std of the dataset
    """
    
    def __init__(self, metric: Metric, mean: list=None, std: list=None):
        super().__init__()
        self.metric = metric
        if mean is None:
            mean = [0]
        if std is None:
            std = [1]
        self.mean = mean
        self.std = std
        if len(mean) == 1:
            self.mean = mean * 3
        if len(std) == 1:
            self.std = std * 3

    def preprocess(self, x, y):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.shape[1] == 1:
            y = y.repeat(1, 3, 1, 1)
        x = denormalize_image(x, self.mean, self.std)
        y = denormalize_image(y, self.mean, self.std)
        return x, y

    @torch.no_grad()
    def forward(self, x, y):
        x, y = self.preprocess(x, y)
        return self.metric(x, y).item()


class SSIM(ReconstructionMetric):
    def __init__(self, mean: list=None, std: list=None):
        super().__init__(StructuralSimilarityIndexMeasure(data_range=1.0), mean, std)


class MS_SSIM(ReconstructionMetric):
    def __init__(self, mean: list=None, std: list=None):
        super().__init__(MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0), mean, std)


class LPIPS(ReconstructionMetric):
    def __init__(self, mean: list=None, std: list=None):
        super().__init__(LearnedPerceptualImagePatchSimilarity(net_type="vgg"), mean, std)
