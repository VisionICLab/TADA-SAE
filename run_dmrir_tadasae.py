import models.swapping_autoencoder as sae
from training.pipelines.ssl_pipelines import SAEDMRIRPipeline
from training.logging.loggers import Logger
from inference.pipelines.tadasae import SymmetryClassifierPipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np
from datasets.dmrir_dataset import DMRIRLeftRightDataset
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from inference.metrics import classification_report
from argparse import ArgumentParser
from tqdm import trange


TADASAE_EXPERIMENTS = ['tadasae_svm', 'tadasae_linear']

class TADASAEExperiment:
    def __init__(self):
        main_parser = ArgumentParser()
        main_parser.add_argument(
            '--experiment',
            type=str,
            default='tadasae_svm',
            choices=TADASAE_EXPERIMENTS,
            help=f'Experiment choice, one of {TADASAE_EXPERIMENTS}'
        )
        
        main_parser.add_argument(
            '--checkpoint',
            type=str,
            required=False,
            help='A path to the .pth file of the trained SAE model'
        )
        
        main_parser.add_argument('--test_only', action="store_true")
           
        self.training_pipeline = SAEDMRIRPipeline(main_parser)
        self.training_pipeline.init_pipeline("./configs/tadasae_dmrir.yaml")
        self.config = self.training_pipeline.get_config()

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

        self.trainer = self.training_pipeline.prepare_trainer(encoder, generator, str_projectors, discriminator, cooccur, Logger(self.config))
        if self.config['checkpoint'] is not None:
            self.trainer.load_state(self.config['checkpoint'])
        
        classifier = SVC(probability=True) if self.config['experiment'] == 'tadasae_svm' else MLPClassifier(hidden_layer_sizes=[])
        self.inference_pipeline = SymmetryClassifierPipeline(self.trainer.enc_ema, RobustScaler(), classifier, self.config['device'])
        
        self.preprocessing = A.Compose([
            A.Resize(self.config['input_size'][1], self.config['input_size'][2]),
            A.Normalize(self.config['mean'], self.config['std']),
            ToTensorV2()
        ], additional_targets={"image0": "image", "mask0": "mask"})
        
        
    def run(self):
        normal_loader, val_loader = self.training_pipeline.prepare_data(self.preprocessing)
        self.training_pipeline.run(self.trainer, normal_loader, val_loader)
    
    def test(self, seeds=1):
        y_n_preds = []
        y_a_preds = []
        
        train_normal_path = os.path.join(self.config['data_root'], self.config['normal_dir_train'])
        train_anomalous_path = os.path.join(self.config['data_root'], self.config['anomalous_dir_train'])
        test_normal_path = os.path.join(self.config['data_root'], self.config['normal_dir_test'])
        test_anomalous_path = os.path.join(self.config['data_root'], self.config['anomalous_dir_test'])

        normal_ds = DMRIRLeftRightDataset(train_normal_path, self.preprocessing, return_mask=False, flip_align=True)
        anomalous_ds = DMRIRLeftRightDataset(train_anomalous_path, self.preprocessing, return_mask=False, flip_align=True)
        # normal_ds_test = DMRIRLeftRightDataset(test_normal_path, self.preprocessing, return_mask=False, flip_align=True)
        # anomalous_ds_test= DMRIRLeftRightDataset(test_anomalous_path, self.preprocessing, return_mask=False, flip_align=True)
        
        # TODO: Implement K-Fold
        for i in trange(seeds, desc=f'Testing classification over {seeds} seeds'):
            
           # np.random.seed(i)
            normal_ds_train, normal_ds_test = normal_ds.split(0.5)
            anomalous_ds_train, anomalous_ds_test = anomalous_ds.split(0.5)  

            self.inference_pipeline.fit_from_dataset(normal_ds_train, anomalous_ds_train)
            (y_normal_pred, _), (y_anomalous_pred, _) = (
                self.inference_pipeline.evaluate_dataset(normal_ds_test, anomalous_ds_test)
            )
            y_n_preds.append(y_normal_pred[:,1]) 
            y_a_preds.append(y_anomalous_pred[:,1])
            self.inference_pipeline.reset()
        classification_report(y_n_preds, y_a_preds)


if __name__ == '__main__':
    experiment = TADASAEExperiment()
    if not experiment.config['test_only']:
        experiment.run()
    experiment.test(1)