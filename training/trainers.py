import torch
from tqdm import tqdm, trange
from abc import ABCMeta
import numpy as np
import os
from inference import metrics
from training.logging.visualisations import visualize_reconstructions
from training.logging.loggers import Logger
import torchmetrics


class Trainer(metaclass=ABCMeta):
    """
    Base class for all trainers with logging and saving functionality

    Args:
        model: model to train
        optimizer: optimizer to use
        loss_fn: loss function to use,
        metrics: metrics to use, in the form of a dictionary of metrics
        config: configuration dictionary
        scheduler: learning rate scheduler to use
    """

    def __init__(
        self, model, optimizer, loss_fn, config, logger: Logger, scheduler=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.logger = logger
        self.scheduler = scheduler
        self.current_epoch = 0

    def train_step(self, x):
        x = x.to(self.config["device"])
        self.optimizer.zero_grad()
        outputs = self.model(x)
        train_loss = self.loss_fn(outputs, x)
        train_loss.backward()
        self.optimizer.step()
        return train_loss.item(), outputs

    @torch.no_grad()
    def eval_step(self, x):
        x = x.to(self.config["device"])
        outputs = self.model(x)[0]
        eval_loss = self.loss_fn(outputs, x)
        return eval_loss.item(), outputs, x

    def train(self, train_loader):
        """
        Trains a reconstruction model for one epoch

        Args:
            train_loader (DataLoader): training data loader
        """
        self.model.train()
        progress_bar = tqdm(train_loader, desc="Training", colour="green")
        for x in progress_bar:
            train_loss, _ = self.train_step(x)
            self.logger.register_log({"train_loss": train_loss})
            progress_bar.set_postfix({"Loss": train_loss})

    def evaluate(self, val_loader):
        self.model.eval()
        progress_bar = tqdm(val_loader, desc="Evaluating reconstruction", colour="cyan")
        for x in progress_bar:
            val_loss, _, _ = self.eval_step(x)
            self.logger.register_log({"eval_loss": val_loss})
            progress_bar.set_postfix({"Loss": val_loss})

    def update_scheduler(self):
        """
        Updates the scheduler based on a given metric
        """
        if self.scheduler is not None:
            metric = (
                None
                if "metric" not in self.config["scheduler"]
                else self.config["scheduler"]["metric"]
            )
            if metric is None or metric not in self.logger.compiled_logs:
                self.scheduler.step()
            else:
                self.scheduler.step(self.logger.compiled_logs[metric])

    def save_state(self, path):
        """
        Saves the state of the trainer to the specified path

        Args:
            path (str): path to which the state will be saved
        """
        dict_to_save = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.scheduler is not None:
            dict_to_save["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(
            dict_to_save,
            path,
        )

    def load_state(self, path):
        """
        Loads the state of the trainer from the specified path to the trainer

        Args:
            path (str): path to load the state from
        """
        checkpoint = torch.load(path, map_location=self.config["device"])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]

    def _make_checkpoint(self, timestamp=0):
        checkpoint_path = os.path.join(
            self.config["log_dir"], "checkpoints", self.config["run_name"]
        )
        if timestamp % self.config["checkpoint_each"] == 0:
            self.save_state(f"{checkpoint_path}_{timestamp}.pth")

    def fit(self, train_loader, val_loader):
        while self.current_epoch < self.config["epochs"]:
            self.current_epoch += 1
            self.train(train_loader)
            self.evaluate(val_loader)
            self.logger.compile_logs()
            self.update_scheduler()
            self._make_checkpoint(self.current_epoch)

            self.logger.log(self.current_epoch)


class ReconstructionTrainer(Trainer):
    """
    A trainer for reconstruction models, such as autoencoders and diffusion models
    """

    def __init__(self, model, optimizer, loss_fn, config, logger, scheduler=None):
        super().__init__(model, optimizer, loss_fn, config, logger, scheduler)
        # self.metrics = {"reconstruction": {}}
        # for metric in config["metrics"]["reconstruction"]:
        #     self.metrics["reconstruction"][metric] = getattr(
        #         reconstruction, metric
        #     )().to(config["device"])

    def _make_visualizations(self, dataset, size=5, title="Reconstructions"):
        random_sampes_idx = np.random.randint(len(dataset), size=size)
        random_samples = []
        random_recons = []
        for i in random_sampes_idx:
            random_sample = dataset[i].unsqueeze(0).to(self.config["device"])
            recons = self.model(random_sample)[0]
            random_recons.append(recons.detach().squeeze(0).cpu())
            random_samples.append(random_sample.detach().squeeze(0).cpu())
        random_recons = torch.stack(random_recons)
        random_samples = torch.stack(random_samples)
        _, fig = visualize_reconstructions(
            random_samples, random_recons, title=title, with_delta=True
        )
        return fig

    @torch.no_grad()
    def evaluate(self, val_loader, with_visualizations=True):
        """
        Evaluates a reconstruction model for one epoch of a validation set

        Args:
            val_loader (DataLoader): validation data loader
        """
        self.model.eval()

        progress_bar = tqdm(val_loader, desc="Evaluating reconstruction", colour="cyan")

        for x in progress_bar:
            val_loss, outputs, gt = self.eval_step(x)

            for metric in self.config["metrics"]["reconstruction"]:
                self.logger.register_log(
                    {metric: self.metrics["reconstruction"][metric](outputs, gt)}
                )

            self.logger.register_log({"eval_loss": val_loss})
            progress_bar.set_postfix({"Loss": val_loss})

        if with_visualizations and self.current_epoch % self.config["log_each"] == 0:
            f = self._make_visualizations(
                val_loader.dataset,
                title=f"Reconstruction at epoch {self.current_epoch}",
            )
            self.logger.register_visualization("reconstruction", f)


class SupervisedTrainer(Trainer):
    def __init__(self, model, optimizer, loss_fn, config, logger, scheduler=None):
        super().__init__(model, optimizer, loss_fn, config, logger, scheduler)
        self.metrics = {
            "acc": torchmetrics.Accuracy(task="binary").to(config["device"]),
            "f1": torchmetrics.F1Score(task="binary").to(config["device"]),
            "precision": torchmetrics.Precision(task="binary").to(config["device"]),
            "recall": torchmetrics.Recall(task="binary").to(config["device"]),
            "auroc": torchmetrics.AUROC(task="binary").to(config["device"]),
            'auprc': torchmetrics.AveragePrecision(task="binary").to(config["device"]),
            "specificity": torchmetrics.Specificity(task="binary").to(config["device"]),
        }

    def train_step(self, x, y):
        x = x.to(self.config['device'])
        y = y.to(self.config["device"]).unsqueeze(1)
        self.optimizer.zero_grad()
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y.float())
        loss.backward()
        self.optimizer.step()
        return loss.item(), y_hat
    
    @torch.no_grad()
    def eval_step(self, x, y):
        x = x.to(self.config['device'])
        y = y.to(self.config["device"]).unsqueeze(1)
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y.float())
        return loss.item(), y_hat, y
    
    def train(self, train_loader):
        self.model.train()
        progress_bar = tqdm(train_loader, desc="Training", colour="green")
        for x, y in progress_bar:
            train_loss, _ = self.train_step(x, y)
            self.logger.register_log({"train_loss": train_loss})
            progress_bar.set_postfix({"Loss": train_loss})
            
    @torch.no_grad()
    def evaluate(self, val_loader, task='eval'):
        assert task in ['eval', 'test'], "Task must be either 'eval' or 'test'"
        self.model.eval()
        progress_bar = tqdm(val_loader, desc="Evaluating", colour="cyan" if task == 'eval' else "yellow")
        for x, y in progress_bar:
            val_loss, y_hat, y = self.eval_step(x, y)
            for metric in self.metrics:
                self.logger.register_log({f"{task}_{metric}": self.metrics[metric](y_hat, y)})
            self.logger.register_log({f"{task}_loss": val_loss})
            progress_bar.set_postfix({"Loss": val_loss})
            
    def fit(self, train_loader, val_loader, test_loader=None):
        while self.current_epoch < self.config["epochs"]:
            self.current_epoch += 1
            self.train(train_loader)
            self.evaluate(val_loader, task='eval')
            if test_loader is not None:
                self.evaluate(test_loader, task='test')
            self.logger.compile_logs()
            self.update_scheduler()
            self._make_checkpoint(self.current_epoch)
            self.logger.log(self.current_epoch)       
    