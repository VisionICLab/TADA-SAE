import numpy as np
import cv2
from torch.utils.data import Dataset
import os
from copy import deepcopy


class DMRIRDataset(Dataset):
    """
    A dataset class which is used to load
    samples from DMRIR dataset given formatting instructions
    passed as parameters.
    
    Args:
        root (str): The root directory of the dataset
        transforms (albumentations.Compose): A composition of albumentations transforms
        side (str): The side of the breast to load. One of ['left', 'right', 'any', 'both']
        return_mask (bool): Whether to return the mask along with the image
        apply_mask (bool): Whether to apply the mask to the image. If False, the side attribute has no effect.
        flip_align (bool): Whether to flip the image and mask horizontally, which ensures all breast images face the same direction
    """

    def __init__(
        self,
        root,
        transforms=None,
        side="both",
        return_mask=True,
        apply_mask=True,
        flip_align=True,
    ):
        self.root = root
        assert side in [
            "left",
            "right",
            "any",
            "both",
        ], "side must be one of ['left', 'right', 'any', 'both']"
        self.side = side
        self.transforms = transforms
        self.return_mask = return_mask
        self.apply_mask = apply_mask
        self.flip_align = flip_align

        self.files = {}
        for name, _, files in os.walk(self.root):
            for f in files:
                if ".png" in f or ".txt" in f:
                    p_num = f.split("PAC_")[1].split("_")[0]
                    if p_num not in self.files:
                        self.files[p_num] = {
                            "matrix": [],
                            "left_mask": [],
                            "right_mask": [],
                        }

                    if name.endswith("matrices"):
                        self.files[p_num]["matrix"].append(os.path.join(name, f))

                    if name.endswith("masks"):
                        side = f.split("_")[-1].split(".")[0] + "_mask"
                        self.files[p_num][side].append(os.path.join(name, f))

    def __len__(self):
        total_len = 0
        for p in self.files:
            total_len += len(self.files[p]["matrix"])
        if self.side == "any":
            total_len *= 2
        return total_len

    def set_side(self, side):
        assert side in [
            "left",
            "right",
            "any",
            "both",
        ], "side must be one of ['left', 'right', 'any', 'both']"
        self.side = side

    def read_tmp_matrix(self, f):
        """
        Read a matrix file and normalize it to [0, 255]
        
        Args:
            f (str): The path to the matrix file
        """
        im = np.loadtxt(f)
        im = (im - np.min(im)) / (np.max(im) - np.min(im))
        im = np.array(im * 255, dtype=np.uint8)
        return im

    def __getitem__(self, idx):
        mul = 2 if self.side == "any" else 1
        for p_id in self.files:
            if idx < len(self.files[p_id]["matrix"]) * mul:
                idx_im = idx // mul

                im_f = self.files[p_id]["matrix"][idx_im]
                if self.side == "left":
                    mask_f = self.files[p_id]["left_mask"][idx]
                elif self.side == "right":
                    mask_f = self.files[p_id]["right_mask"][idx]
                elif self.side == "any":
                    if idx % 2 == 0:
                        mask_f = self.files[p_id]["left_mask"][idx_im]
                    else:
                        mask_f = self.files[p_id]["right_mask"][idx_im]

                img = self.read_tmp_matrix(im_f)

                if self.side == "both":
                    mask_l = cv2.imread(
                        self.files[p_id]["left_mask"][idx], cv2.IMREAD_GRAYSCALE
                    )
                    mask_r = cv2.imread(
                        self.files[p_id]["right_mask"][idx], cv2.IMREAD_GRAYSCALE
                    )
                    mask = mask_l + mask_r
                    img = cv2.resize(img, mask_l.shape[::-1])
                else:
                    mask = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
    
                img = cv2.resize(img, mask.shape[::-1])

                if self.apply_mask:
                    img = img * (mask > 0)
                if self.flip_align and (
                    (idx % 2 == 1 and self.side == "any") or self.side != "any"
                ):
                    img = cv2.flip(img, 1)
                    mask = cv2.flip(mask, 1)

                if self.transforms is not None:
                    augmented = self.transforms(image=img, mask=mask)
                    img = augmented["image"]
                    mask = augmented["mask"]
                    mask = mask.squeeze(0) > 0
                if self.return_mask:
                    return img, mask
                return img
            else:
                idx -= len(self.files[p_id]["matrix"]) * mul
                
    def split(self, train_ratio=0.8):
        """
        Randomly split the dataset into training and validation sets by patient.
        
        Args:
            train_ratio (float): The ratio of the dataset to use for training
            
        Returns:
            DMRIRDataset: A training dataset
            DMRIRDataset: A validation dataset
        """
        # select random patients
        patients = list(self.files.keys())
        np.random.shuffle(patients)
        n_train = int(train_ratio * len(patients))
        train_patients = patients[:n_train]
        val_patients = patients[n_train:]

        train_files = {}
        val_files = {}
        for p in train_patients:
            train_files[p] = self.files[p]
        for p in val_patients:
            val_files[p] = self.files[p]

        train_dataset = deepcopy(self)
        train_dataset.files = train_files

        val_dataset = deepcopy(self)
        val_dataset.files = val_files

        return train_dataset, val_dataset


class DMRIRLeftRightDataset(DMRIRDataset):
    """
    A DMRIRMatrixDataset dataset subclass to return paired left-right breast images
    belonging to the same patient
    """
    
    def __init__(self, root, transforms=None, return_mask=True, apply_mask=True, flip_align=True):
        super().__init__(root, side="both", transforms=transforms, return_mask=return_mask, apply_mask=apply_mask, flip_align=flip_align)
    
    def _get_image_mask_from_idx(self, patient_idx, idx):
        """
        Get the image and mask from the dataset given the patient index and the index of the image
        
        Args:
            patient_idx (int): The patient index
            idx (int): The index of the image
            
        Returns:
            np.array: The image
            np.array: The left mask
            np.array: The right mask
        """
        im_f = self.files[patient_idx]["matrix"][idx]
        mask_l_f = self.files[patient_idx]["left_mask"][idx]
        mask_r_f = self.files[patient_idx]["right_mask"][idx]   
        img = self.read_tmp_matrix(im_f)
        mask_l = cv2.imread(mask_l_f, cv2.IMREAD_GRAYSCALE)
        mask_r = cv2.imread(mask_r_f, cv2.IMREAD_GRAYSCALE)
        
        img = cv2.resize(img, mask_l.shape[::-1])
        
        return img, mask_l, mask_r
                
    def __getitem__(self, idx):
        for p_id in self.files:
            if idx < len(self.files[p_id]["matrix"]):
                img, mask_l, mask_r = self._get_image_mask_from_idx(p_id, idx)
                img_l, img_r = img, img
                if self.apply_mask:
                    img_l = img * (mask_l > 0)
                    img_r = img * (mask_r > 0)
                
                if self.flip_align:
                    img_r = cv2.flip(img_r, 1)
                    mask_r = cv2.flip(mask_r, 1)                    

                if self.transforms is not None:
                    augmented = self.transforms(
                        image=img_l, image0=img_r, mask=mask_l, mask0=mask_r
                    )
                    img_l = augmented["image"]
                    img_r = augmented["image0"]
                    mask_l = augmented["mask"].float() / 255
                    mask_r = augmented["mask0"].float() / 255
                if self.return_mask:
                    return (img_l, mask_l), (img_r, mask_r)
                return img_l, img_r
            else:
                idx -= len(self.files[p_id]["matrix"])


class LabeledDMRIRDataset(DMRIRDataset):
    """
    A DMRIRDataset subclass which returns images along with their class labels
    
    Args:
        root (str): The root directory of the dataset
        class_label (int): The class label to assign to the images
        transforms (albumentations.Compose): A composition of albumentations transforms
        apply_mask (bool): Whether to apply the mask to the image
    """
    def __init__(self, root, class_label, transforms=None, apply_mask=True):
        super().__init__(root, transforms, side='both', return_mask=False, apply_mask=apply_mask, flip_align=False)
        self.class_label = class_label
    
    def __getitem__(self, idx):
        return super().__getitem__(idx), self.class_label