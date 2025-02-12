from argparse import ArgumentParser
from experiment import AbstractExperiment
from training.pipelines.ssl_pipelines import AEDMRIRPipeline, SAEDMRIRPipeline
from enum import Enum

class ModelTypes(Enum):
    LSAE='lsae'
    AE='ae'


class TADASAEAblationExperiments(AbstractExperiment):
    def __init__(self):
        super().__init__(['ae_full_im', 'ae_left_right', 'lsae_left_right', 'lsae_full_im'])
        experiment_name = vars(self.main_parser.parse_args())['experiment']
        model_type = experiment_name.split('_')[0]
        sym = sum(experiment_name.split('_')[1:-1])
        
        if model_type == ModelTypes.AE:
            self.training_pipeline = AEDMRIRPipeline(self.main_parser)
        else:
            self.training_pipeline = SAEDMRIRPipeline(self.main_parser)
        
        #self.inference_pipeline = SymmetryClassifierPipeline(self.trainer.enc_ema, RobustScaler(), classifier, self.config['device'])
