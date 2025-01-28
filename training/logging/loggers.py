import os
import wandb
from pathlib import Path
import matplotlib.pyplot as plt
from copy import deepcopy
from .utils import tensor2image


class Logger:
    DEFAULT_CONFIG = {
        "epochs": 0,
        "log_each": 1,
        "checkpoint_each": 1,
        "entity": None,
        "project_name": None,
        "project_group": None,
        "run_name": None,
        "wandb": False,
        "log_dir": "./"
    }

    def __init__(self, config=None):
        """
        A logger class to handle logging of training metrics and visualizations, working as a context manager
        
        Args:
            config (dict): A dictionary containing the training configuration:
                "log_dir": <directory for image logging>
                "epochs": <number of epochs, default 0>
                "steps": <number of train steps, optional>
                "log_each": <logging occurence>
                "checkpoint_each": <model saving occurence>
                "wandb": <True or False>,
                "entity": <Weights and Biases entity>,
                "project_name": <Weights and Biases project name>,
                "project_group": <Weights and Biases project group>,
                "run_name": <run name for Weights and Biases + logging>
        """
        self.logs = {}
        self.config = config

        if self.config is None:
            self.config = Logger.DEFAULT_CONFIG

        self.log_dir = self.config["log_dir"]
        self.visualization = {}
        self.tensors = {}
        self.compiled_logs = {}
        self.total_steps = (
            self.config["steps"]
            if "steps" in self.config.keys()
            else self.config["epochs"]
        )
        self.log_each = self.config["log_each"]
        self.with_wandb = self.config["wandb"]
        
        if self.config["log_dir"]:
            Path(os.path.join(self.config["log_dir"], "checkpoints")).mkdir(
                parents=True, exist_ok=True
            )

        if self.with_wandb:
            wandb.init(
                entity=self.config["entity"],
                project=self.config["project_name"],
                group=self.config["project_group"],
                name=self.config["run_name"] if "run_name" in self.config else None,
                config=self.config,
            )



    def print_logs(self, current_step):
        print(f"===== {current_step}/{self.total_steps} =====")
        for k, v in self.compiled_logs.items():
            print(f"{k}: {v:.6f}")

    def save_figures(self, current_step):
        for name, fig in self.visualization.items():
            fig.savefig(
                os.path.join(self.log_dir, f"{name}_{current_step}.png"),
                bbox_inches="tight",
            )
            fig.show()

        for name, t in self.tensors.items():
            image = tensor2image(t)
            plt.imsave(os.path.join(self.log_dir, f"{name}_{current_step}.png"), image)

    def register_visualization(self, name, fig):
        self.visualization[name] = deepcopy(fig)

    def register_tensor(self, name, tensor):
        self.tensors[name] = tensor

    def register_log(self, log_dict, **kwargs):
        logs = {**log_dict, **kwargs}
        for key, value in logs.items():
            if key in self.logs:
                self.logs[key]["value"] += value
                self.logs[key]["it"] += 1
            else:
                self.logs[key] = {"value": value, "it": 1}

    def compile_logs(self):
        for key, value in self.logs.items():
            self.compiled_logs[key] = value["value"] / value["it"]

    def reset_logs(self):
        self.logs = {}
        self.visualization = {}
        self.tensors = {}

    def log(self, current_epoch=0, clear_previous=True):
        assert (
            len(self.compiled_logs.keys()) > 0
        ), "No logs to print, call logger.compile_logs() first"
        if clear_previous:
            os.system("clear")
        if current_epoch % self.log_each == 0:
            self.print_logs(current_epoch)
            self.save_figures(current_epoch)

            if self.with_wandb:
                wandb_fig_vis = {}
                wandb_img_vis = {}
                for name, fig in self.visualization.items():
                    fig.tight_layout()
                    wandb_fig_vis[name] = wandb.Image(fig)
                for name, img in self.tensors.items():
                    wandb_img_vis[name] = wandb.Image(img)
                wandb.log({**self.compiled_logs, **wandb_fig_vis, **wandb_img_vis})
        plt.clf()
        plt.close("all")
        self.reset_logs()

    def shutdown(self):
        print("===== Training Finished =====")
        if self.with_wandb:
            wandb.finish()

    def __exit__(self, exc_type, exc_value, traceback):
        self.shutdown()
