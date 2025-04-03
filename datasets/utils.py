import random
import copy
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold, GroupKFold

def split(dataset, ratio=0.8, shuffle=True):
    """
    Splits two datasets into two new datasets, where the first dataset is a ratio of the original dataset
    and the second dataset is the remaining ratio of the original dataset.

    Args:
        dataset_a (Dataset): dataset to split
        ratio (float): ratio of the first dataset to the original dataset

    Returns:
        (Dataset, Dataset): two new datasets, where the first dataset is a ratio of the original dataset

    """
    dataset_a = copy.deepcopy(dataset)
    dataset_b = copy.deepcopy(dataset)
    file_paths = dataset.file_paths
    if shuffle:
        random.shuffle(file_paths)
    split_idx = int(len(file_paths) * ratio)
    file_paths_a = file_paths[:split_idx]
    file_paths_b = file_paths[split_idx:]
    dataset_a.file_paths = file_paths_a
    dataset_b.file_paths = file_paths_b
    return dataset_a, dataset_b


class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        """
        https://gist.github.com/MFreidank/821cc87b012c53fade03b0c7aba13958
        """
        super().__init__(*args, **kwargs)
        # Initialize an iterator over the dataset.
        self.dataset_iterator = super().__iter__()

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            # Dataset exhausted, use a new fresh iterator.
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


def apply_cv2_colormap(img_mono, cmap):
    img_colored = cv2.applyColorMap(img_mono, cmap)
    img_colored = cv2.cvtColor(img_colored, cv2.COLOR_BGR2RGB)
    return img_colored


def denormalize_image(im, mean, std):
    denorm = (
        im * torch.tensor(std, device=im.device)[:, None, None]
        + torch.tensor(mean, device=im.device)[:, None, None]
    )
    return denorm.clamp(0, 1).float()


class KFoldDatasetWrapper(Dataset):
    def __init__(self, dataset, n_splits=10):
        self.dataset=dataset
        self.n_splits=n_splits
        kf = KFold(self.n_splits)
    
    def __getitem__(self, index):
        return super().__getitem__(index)
    
    def __len__(self):
        return len(self.dataset)
        
        