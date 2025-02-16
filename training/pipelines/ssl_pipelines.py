from argparse import ArgumentParser
import os
import torch
from torch.nn import DataParallel
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets.dmrir_dataset import DMRIRMatrixDataset
from datasets.utils import InfiniteDataLoader
import numpy as np
from torch.utils.data import random_split
from training.pipelines.pipeline import AbstractPipeline, TrainMode
import yaml
from models.ema import EMA
from models.utils import count_parameters
from training.sae import SAETrainer
from training.trainers import ReconstructionTrainer
from preprocessing.ada_aug import ADAAugment
import models.swapping_autoencoder as sae
from models.ablated_models import ConvEncoder, ConvDecoder
from training.logging.loggers import Logger



class AEDMRIRPipeline(AbstractPipeline):
    """
    A pipeline for training on DMRIR dataset with convolutional autoencoder
    """
    def __init__(self, main_parser=...):
        super().__init__(main_parser)
        
    
    def init_pipeline(self, config_path=None):
        """
        A function initializing module state for pipeline
        """
        args_dict = vars(self.parser.parse_args())
        self.config_path = (
            args_dict["config_path"]
            if args_dict["config_path"] is not None
            else config_path
        )
        assert self.config_path is not None, "Config path must be specified"

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        for k, v in args_dict.items():
            if k in self.config.keys() and v is not None:
                self.config[k] = v

        if not os.path.exists(self.config["log_dir"]):
            os.mkdir(self.config["log_dir"])

        if "seed" in self.config:
            self.set_seed(self.config["seed"])
            
    def prepare_data(self, transforms=..., mode=TrainMode.FULL):
        if transforms is None:
            transforms = A.Compose(
                [
                    A.Resize(self.config["input_size"][1], self.config["input_size"][-1]),
                    A.Normalize(self.config["mean"], self.config["std"]),
                    ToTensorV2(),
                ],
                additional_targets={"image0": "image", "mask0": "mask"},
            )

        normal_path = os.path.join(self.config["data_root"], self.config["normal_dir_train"])
        ano_path = os.path.join(self.config["data_root"], self.config["anomalous_dir_train"])

        side = 'both' if mode == TrainMode.FULL.value else 'any'
        normal_ds = DMRIRMatrixDataset(normal_path, transforms, side, return_mask=False)
        ano_ds = DMRIRMatrixDataset(ano_path, transforms, side, return_mask=False)

        normal_train_ds, ano_val_ds = random_split(
            normal_ds,
            [int(len(normal_ds) * 0.9), len(normal_ds) - int(len(normal_ds) * 0.9)],
        )
        
        ano_train_ds, ano_eval_ds = random_split(
            ano_ds, [int(len(ano_ds) * 0.8), len(ano_ds) - int(len(ano_ds) * 0.8)]
        )

        train_ds = torch.utils.data.ConcatDataset([normal_train_ds, ano_train_ds])
        val_ds = torch.utils.data.ConcatDataset([ano_val_ds, ano_eval_ds])

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=self.config["num_workers"],
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            drop_last=True,
            num_workers=self.config["num_workers"],
        )
        return train_loader, val_loader
    
    def _prepare_training(self):
        enc = ConvEncoder(512, 32, 1).to(self.config['device'])  # TODO: CHange for custom dimensions from config file
        dec = ConvDecoder(512, 32, 1).to(self.config['device'])
        return enc, dec, *super()._prepare_training(list(enc.parameters())+list(dec.parameters()))

    def prepare_trainer(self):
        enc, gen, optimizer, loss_fn, scheduler = self._prepare_training()
        print(f"Encoder: {count_parameters(enc)}")
        print(f"Generator: {count_parameters(gen)}")
        return ReconstructionTrainer(enc, gen, optimizer, loss_fn, self.config, Logger(self.config), scheduler)

    def run(self, trainer, train_loader, val_loader):
        return trainer.fit(train_loader, val_loader)

class SAEDMRIRPipeline(AEDMRIRPipeline):
    """
    A pipeline for training on DMRIR dataset with Swapping Autoencoder
    """
    def __init__(self, main_parser=...):
        super().__init__(main_parser)
        
    # def prepare_data(self, transforms=None, **kwargs):
    #     return super().prepare_data(transforms, TrainMode.LR)

    def _prepare_training(self):
        CHANNELS = self.config["channels"]
        STRUCTURE_CHANNELS = self.config["struct_channels"]
        TEXTURE_CHANNELS = self.config["text_channels"]

        encoder = sae.encoders.PyramidEncoder(
            CHANNELS,
            structure_channel=STRUCTURE_CHANNELS,
            texture_channel=TEXTURE_CHANNELS,
            gray=True,
        ).to(self.config["device"])

        generator = sae.generators.Generator(
            CHANNELS,
            structure_channel=STRUCTURE_CHANNELS,
            texture_channel=TEXTURE_CHANNELS,
            gray=True,
        ).to(self.config["device"])

        str_projectors = sae.layers.MultiProjectors(
            [CHANNELS, CHANNELS * 2, CHANNELS * 8], use_mlp=True
        ).to(self.config["device"])

        discriminator = sae.discriminators.Discriminator(
            self.config["input_size"][-1], channel_multiplier=1, gray=True
        ).to(self.config["device"])

        cooccur = sae.discriminators.CooccurDiscriminator(
            CHANNELS, size=self.config["input_size"][-1] * self.config["max_patch_size"], gray=True
        ).to(self.config["device"])
        
        return encoder, generator, str_projectors, discriminator, cooccur
    
    def prepare_trainer(self):
        enc, gen, str_proj, disc, cooccur = self._prepare_training()
        print(f"Encoder: {count_parameters(enc)}")
        print(f"Generator: {count_parameters(gen)}")
        print(f"StrProjectors: {count_parameters(str_proj)}")
        print(f"Discriminator: {count_parameters(disc)}")
        print(f"Cooccur: {count_parameters(cooccur)}")

        enc_ema = EMA(enc, power=3 / 4).to(self.config["device"])
        gen_ema = EMA(gen, power=3 / 4).to(self.config["device"])

        size = self.config["input_size"][1:]
        mean = np.array(self.config["mean"])
        std = np.array(self.config["std"])

        ada_aug = ADAAugment(
            ada_target=0.8, ada_step=1e-4, max_proba=0.8, size=size, mean=mean, std=std
        ).to(self.config["device"])

        if torch.cuda.device_count() > 1:
            enc = DataParallel(enc)
            gen = DataParallel(gen)
            str_proj = DataParallel(str_proj)
            disc = DataParallel(disc)
            cooccur = DataParallel(cooccur)

        trainer = SAETrainer(
            enc,
            gen,
            str_proj,
            disc,
            cooccur,
            enc_ema,
            gen_ema,
            ada_aug,
            self.config,
            Logger(self.config)
        )
        return trainer


    def run(self, trainer, normal_loader, val_loader):
        trainer.fit(normal_loader, val_loader)
