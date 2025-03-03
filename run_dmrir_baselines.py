from functools import partial
from training.pipelines.supervised_pipelines import DMRIRSupervisedPipeline
from models.supervised.convolutional import ConvSmall, ResNet18, InceptionV3
from training.logging.loggers import Logger
from experiment import AbstractExperiment


MODELS = {
    'conv_small': partial(ConvSmall, z_dim=128, c_hid=32, c_in=1),
    'resnet_18': ResNet18,
    'inception_v3': InceptionV3
}

class DMRIRBaselineExperiments(AbstractExperiment):
    def __init__(self):
        super().__init__(['conv_small', 'resnet_18', 'inception_v3'])
        self.pipeline = DMRIRSupervisedPipeline(self.main_parser)
        self.pipeline.init_pipeline("./configs/supervised_dmrir.yaml")
    
    def run(self):
        train_loader, val_loader, test_loader = self.pipeline.prepare_data()
    
        model = MODELS[self.experiment]().to(self.pipeline.config['device'])
        with Logger(self.pipeline.config) as logger:
            trainer = self.pipeline.prepare_trainer(model, logger)
            self.pipeline.run(trainer, train_loader, val_loader, test_loader)        
        

if __name__ == '__main__':
    experiments = DMRIRBaselineExperiments()
    experiments.run()