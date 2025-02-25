import os
from tqdm import trange
from experiment import AbstractExperiment
from training.pipelines.ssl_pipelines import AEDMRIRPipeline, TADASAEDMRIRPipeline
from enum import Enum
from inference.pipelines.tadasae import SymmetryClassifierPipeline, AnomalyDetectionPipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from training.pipelines.pipeline import TrainMode
from datasets.dmrir_dataset import DMRIRMatrixDataset, DMRIRLeftRightDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from functools import partial
from inference.metrics import classification_report
import numpy as np


class ModelTypes(Enum):
    LSAE='lsae'
    AE='ae'

class TADASAEAblationExperiments(AbstractExperiment):
    def __init__(self):
        super().__init__(['ae_full_im', 'ae_left_right', 'lsae_left_right', 'lsae_full_im'])
        experiment_name = vars(self.main_parser.parse_args())['experiment']
        model_type = experiment_name.split('_')[0]
        self.mode = '_'.join(experiment_name.split('_')[1:])
        
        if model_type == ModelTypes.AE.value:
            self.training_pipeline = AEDMRIRPipeline(self.main_parser)
            self.training_pipeline.init_pipeline('configs/ae_dmrir.yaml')
        else:
            self.training_pipeline = TADASAEDMRIRPipeline(self.main_parser)
            self.training_pipeline.init_pipeline('configs/tadasae_dmrir.yaml')

        self.config = self.training_pipeline.config
        self.trainer = self.training_pipeline.prepare_trainer()
        
        self.preprocessing = A.Compose(
                [
                    A.Resize(self.config["input_size"][1], self.config["input_size"][-1]),
                    A.Normalize(self.config["mean"], self.config["std"]),
                    ToTensorV2(),
                ],
                additional_targets={"image0": "image", "mask0": "mask"},
            )

    def run(self):
        train_loader, val_loader = self.training_pipeline.prepare_data(transforms=self.preprocessing, mode=self.mode)
        self.training_pipeline.run(self.trainer, train_loader, val_loader)

    def test(self, seeds=1):       
        y_n_preds = []
        y_a_preds = []
        
        if self.mode == TrainMode.FULL:
            PIPELINE_CLASS = AnomalyDetectionPipeline
            DS_CLASS = partial(DMRIRMatrixDataset, transforms=self.preprocessing, side='both', return_mask=False)
        else:
            PIPELINE_CLASS = SymmetryClassifierPipeline
            DS_CLASS = partial(DMRIRLeftRightDataset, transforms=self.preprocessing, return_mask=False)
        
        
            
        inference_pipeline = PIPELINE_CLASS(self.trainer.encoder, RobustScaler(), SVC(probability=True), device=self.config['device'])
        
        normal_ds_train = DS_CLASS(root=os.path.join(self.config['data_root'], self.config['normal_dir_train']))
        anomalous_ds_train = DS_CLASS(root=os.path.join(self.config['data_root'], self.config['anomalous_dir_train']))
        normal_ds_val = DS_CLASS(root=os.path.join(self.config['data_root'], self.config['normal_dir_test']))
        anomalous_ds_val = DS_CLASS(root=os.path.join(self.config['data_root'], self.config['anomalous_dir_test']))
        
        for i in trange(seeds, desc=f'Testing classification over {seeds} seeds'):
            np.random.seed(i)
            inference_pipeline.fit_from_dataset(normal_ds_train, anomalous_ds_train)
            (y_normal_pred, _), (y_anomalous_pred, _) = (
                inference_pipeline.evaluate_dataset(normal_ds_val, anomalous_ds_val)
            )
            y_n_preds.append(y_normal_pred[:,1]) 
            y_a_preds.append(y_anomalous_pred[:,1])
            inference_pipeline.reset()
        classification_report(y_n_preds, y_a_preds)

if __name__ == '__main__':
    experiment = TADASAEAblationExperiments()
    if not experiment.config['test_only']:
        experiment.run()
    experiment.test(1)
