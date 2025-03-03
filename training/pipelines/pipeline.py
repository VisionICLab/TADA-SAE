from abc import ABCMeta, abstractmethod
import os
import yaml
from argparse import ArgumentParser
import torch
from enum import Enum


class TrainMode(Enum):
    LR='left_right'
    FULL='full_im'


OPTIMIZERS = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SGD": torch.optim.SGD,
    "NAdam": torch.optim.NAdam,
}

SCHEDULERS = {
    "OneCycleLR": torch.optim.lr_scheduler.OneCycleLR,
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "StepLR": torch.optim.lr_scheduler.StepLR,
}

LOSS_FUNCTIONS = {
    "L1Loss": torch.nn.L1Loss,
    "MSELoss": torch.nn.MSELoss,
    "BCELoss": torch.nn.BCELoss,
}


class AbstractPipeline(metaclass=ABCMeta):
    def __init__(self, main_parser=ArgumentParser()):
        """
        A base pipeline for training models and running experiments
        """
        self.config = {}
        
        self.parser = main_parser
    
        self.parser.add_argument(
            "--log_dir",
            type=str,
            default=None,
            help="Path to the output directory",
        )

        self.parser.add_argument(
            "--config_path",
            type=str,
            default=None,
            help="Path to the .yaml config file",
        )

        self.parser.add_argument(
            "--data_root",
            type=str,
            default=None,
            help="Path to the data root",
        )

        self.parser.add_argument(
            "--normal_dir",
            type=str,
            default=None,
            help="Path to the normal data",
        )

        self.parser.add_argument(
            "--anomaly_dir",
            type=str,
            default=None,
            help="Path to the anomaly data",
        )

        self.parser.add_argument(
            "-d",
            "--device",
            type=str,
            default=None,
            help="Device to use for training",
        )

        self.parser.add_argument(
            "--seed", type=int, default=None, help="Seed to use for reproducibility"
        )

        self.parser.add_argument(
            "--project_group",
            type=str,
            default=None,
            help="Name of the project group (wandb)",
        )

        self.training_group = self.parser.add_argument_group("Training")

        self.training_group.add_argument(
            "--epochs", type=int, default=None, help="Number of epochs to train for"
        )

        self.training_group.add_argument(
            "--batch_size",
            type=int,
            default=None,
            help="Batch size to use for training",
        )

        self.training_group.add_argument(
            "--checkpoint_each",
            type=int,
            default=None,
            help="Save model every n epochs",
        )

        self.training_group.add_argument(
            "--model_name",
            type=str,
            default=None,
            help="Name of the model to use",
        )

        self.training_group.add_argument(
            "--run_name",
            type=str,
            default=None,
            help="Name of the run (wandb)",
        )

        self.training_group.add_argument(
            "--wandb", action="store_true", help="Whether to use wandb for logging"
        )


    def set_seed(self, seed=None):
        self.config["seed"] = seed
        if seed is not None:
            torch.manual_seed(seed)

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
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        for k, v in args_dict.items():
            if k in self.config.keys() and v is not None:
                self.config[k] = v

        os.makedirs(self.config["log_dir"], exist_ok=True)
        self.set_seed(self.config["seed"])


    def _prepare_training(self, parameters):
        optimizer_class = OPTIMIZERS[self.config["optimizer"]["name"]]
        scheduler_class = (
            SCHEDULERS[self.config["scheduler"]["name"]]
            if "scheduler" in self.config.keys()
            else None
        )
        loss_fn_class = LOSS_FUNCTIONS[self.config["loss_fn"]["name"]]
        
        optimizer = optimizer_class(
            parameters, **self.config["optimizer"]["kwargs"]
        )
        loss_fn = loss_fn_class(**self.config["loss_fn"]["kwargs"]).to(
            self.config["device"]
        )
        if scheduler_class is not None:
            scheduler = scheduler_class(
                optimizer, **self.config["scheduler"]["kwargs"]
            )
        else:
            scheduler = None

        return optimizer, loss_fn, scheduler

    def get_config(self):
        return self.config

    def set_config(self, **kwargs):
        self.config.update(kwargs)

    def save_model(self, trainer):
        trainer.save_model(
            os.path.join(
                self.config["log_dir"],
                f"{self.config['project_group']}_{type(trainer.get_model()).__name__}.pth",
            )
        )

    @abstractmethod
    def prepare_data(self, data_path):
        pass

    @abstractmethod
    def prepare_trainer(self):
        pass

    @abstractmethod
    def run(self):
        pass
