import os
from tqdm import trange
from training.pipelines.ssl_pipelines import TADASAEDMRIRPipeline
from inference.pipelines.tadasae import TextureSymmetryClassifierPipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from datasets.dmrir_dataset import DMRIRLeftRightDataset
from inference.metrics import classification_report
from experiment import AbstractExperiment
import albumentations as A
from albumentations.pytorch import ToTensorV2


class TADASAEExperiment(AbstractExperiment):
    def __init__(self):
        super().__init__(['tadasae'])
    
        self.training_pipeline = TADASAEDMRIRPipeline(self.main_parser)
        self.training_pipeline.init_pipeline("./configs/tadasae_dmrir.yaml")

        self.trainer = self.training_pipeline.prepare_trainer()
        if self.config['checkpoint'] is not None:
            self.trainer.load_state(self.config['checkpoint'])
        
        self.preprocessing = A.Compose(
                [
                    A.Resize(self.config["input_size"][1], self.config["input_size"][-1]),
                    A.Normalize(self.config["mean"], self.config["std"]),
                    ToTensorV2(),
                ],
                additional_targets={"image0": "image", "mask0": "mask"},
            )

        classifier = SVC(probability=True) if self.config['classifier'] == 'svm' else MLPClassifier(hidden_layer_sizes=[])
        self.inference_pipeline = TextureSymmetryClassifierPipeline(self.trainer.enc_ema, RobustScaler(), classifier, self.config['device'])

    @property
    def config(self):
        return self.training_pipeline.config

    def run(self):
        normal_loader, val_loader = self.training_pipeline.prepare_data()
        self.training_pipeline.run(self.trainer, normal_loader, val_loader)
    
    def test(self, seeds=1):
        y_n_preds = []
        y_a_preds = []
        
        train_normal_path = os.path.join(self.config['data_root'], self.config['normal_dir_train'])
        train_anomalous_path = os.path.join(self.config['data_root'], self.config['anomalous_dir_train'])
        test_normal_path = os.path.join(self.config['data_root'], self.config['normal_dir_test'])
        test_anomalous_path = os.path.join(self.config['data_root'], self.config['anomalous_dir_test'])

        normal_ds_train = DMRIRLeftRightDataset(train_normal_path, self.preprocessing, return_mask=False)
        anomalous_ds_train = DMRIRLeftRightDataset(train_anomalous_path, self.preprocessing, return_mask=False)
        normal_ds_test = DMRIRLeftRightDataset(test_normal_path, self.preprocessing, return_mask=False, flip_align=True)
        anomalous_ds_test = DMRIRLeftRightDataset(test_anomalous_path, self.preprocessing, return_mask=False, flip_align=True)
        
        for _ in trange(seeds, desc=f'Testing classification over {seeds} seeds'):
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
    experiment.test(2)