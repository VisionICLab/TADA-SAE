import os
from training.pipelines.pipeline import AbstractPipeline
from training.trainers import SupervisedTrainer
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets.dmrir_dataset import LabeledDMRIRDataset
from torch.utils.data import ConcatDataset, DataLoader
from functools import partial


class DMRIRSupervisedPipeline(AbstractPipeline):
    """
    A pipeline for training baseline supervised models
    as presented in the paper for DMRIR dataset
    """
    def prepare_data(self, with_augmentations=True):
        _, h, w = self.config['input_size']
        tr_pipeline = [
            A.Normalize(self.config["mean"], self.config["std"]),
            ToTensorV2()
        ]
        aug_pipeline = []
        if with_augmentations:
            aug_pipeline = [
                A.Affine(
                    translate_percent=(0.125, 0.25), 
                    rotate=(-30, 30), 
                    scale=(0.8, 1.2), 
                    shear=(-30, 30), 
                    p=0.5
                ),
                A.ElasticTransform(),
                A.GaussianBlur()
            ]
            
        aug_pipeline += tr_pipeline
        train_transforms = A.Compose([A.Resize(h, w)] + aug_pipeline)
        test_transforms = A.Compose([A.Resize(h, w)] + tr_pipeline)
        
        train_n_path = os.path.join(self.config['root'], 'train', self.config['normal_dir'])
        train_a_path = os.path.join(self.config['root'], 'train', self.config['anomaly_dir'])
        
        test_n_path = os.path.join(self.config['root'], 'test', self.config['normal_dir'])
        test_a_path = os.path.join(self.config['root'], 'test', self.config['anomaly_dir'])
        
        train_n_ds, val_n_ds = LabeledDMRIRDataset(train_n_path, 0, train_transforms, apply_mask=False).split()
        train_a_ds, val_a_ds = LabeledDMRIRDataset(train_a_path, 1, train_transforms, apply_mask=False).split()

        test_n_ds = LabeledDMRIRDataset(test_n_path, 0, test_transforms)
        test_a_ds = LabeledDMRIRDataset(test_a_path, 1, test_transforms)
        
        val_n_ds.transforms = test_transforms
        val_a_ds.transforms = test_transforms
        
        train_ds = ConcatDataset([train_n_ds, train_a_ds])
        val_ds = ConcatDataset([val_n_ds, val_a_ds])
        test_ds = ConcatDataset([test_n_ds, test_a_ds])
        dataloader_class = partial(DataLoader, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config['num_workers'])
        
        return dataloader_class(train_ds), dataloader_class(val_ds), dataloader_class(test_ds)
    
    def prepare_trainer(self, model, logger):
        optimizer, loss_fn, scheduler = self._prepare_training(model)
        return SupervisedTrainer(
            model, optimizer, loss_fn, self.config, logger, scheduler
        )

    def run(self, trainer, train_loader, val_loader, test_loader):
        trainer.fit(train_loader, val_loader, test_loader)
