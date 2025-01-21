from argparse import ArgumentParser
from functools import partial
from training.pipelines.supervised_pipelines import DMRIRSupervisedPipeline
from models.supervised.convolutional import ConvSmall, ResNet18, InceptionV3
from training.logging.loggers import Logger

# EXPERIMENTS = {
#     'BASELINE_EXPERIMENTS': ['conv_small', 'resnet_18', 'inception_v3'],
#     'ABL_EXPERIMENTS': ['ae_full_im', 'ae_left_right', 'lsae_left_right', 'lsae_full_im'],
#     'FINAL_EXPERIMENTS':  ['tada_sae']
# }

BASELINE_EXPERIMENTS = ['conv_small', 'resnet_18', 'inception_v3']

MODELS = {
    'conv_small': partial(ConvSmall, z_dim=128, c_hid=32, c_in=1),
    'resnet_18': ResNet18,
    'inception_v3': InceptionV3
}

class DMRIRBaselineExperiments:
    def __init__(self):
        main_parser = ArgumentParser()
        main_parser.add_argument(
            '--experiment',
            type=str,
            default='conv_small',
            choices=BASELINE_EXPERIMENTS,
            help=f'Experiment choice, one of {BASELINE_EXPERIMENTS}'
        )
        self.experiment = vars(main_parser.parse_args())['experiment']
        self.pipeline = DMRIRSupervisedPipeline(main_parser)
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