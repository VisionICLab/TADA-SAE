from experiment import AbstractExperiment
from training.pipelines.ssl_pipelines import AEDMRIRPipeline, SAEDMRIRPipeline
from enum import Enum
from inference.pipelines.tadasae import SymmetryClassifierPipeline, AnomalyDetectionPipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from training.pipelines.pipeline import TrainMode

class ModelTypes(Enum):
    LSAE='lsae'
    AE='ae'


class TADASAEAblationExperiments(AbstractExperiment):
    def __init__(self):
        super().__init__(['ae_full_im', 'ae_left_right', 'lsae_left_right', 'lsae_full_im'])
        experiment_name = vars(self.main_parser.parse_args())['experiment']
        model_type = experiment_name.split('_')[0]
        self.mode = sum(experiment_name.split('_')[1:-1])
        
        if model_type == ModelTypes.AE:
            self.training_pipeline = AEDMRIRPipeline(self.main_parser)
        else:
            self.training_pipeline = SAEDMRIRPipeline(self.main_parser)
            
        self.trainer = self.training_pipeline.prepare_trainer()
    
    def run(self):
        train_loader, val_loader = self.training_pipeline.prepare_data(mode=self.mode)
        self.training_pipeline.run(self.trainer, train_loader, val_loader)
        
    def test(self, seeds=1):
        PIPELINE_CLASS = SymmetryClassifierPipeline if self.mode == TrainMode.LR else AnomalyDetectionPipeline
        inference_pipeline = PIPELINE_CLASS(self.trainer.encoder, RobustScaler(), SVC(probability=True))
    