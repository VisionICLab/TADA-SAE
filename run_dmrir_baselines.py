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

if __name__ == '__main__':
    main_parser = ArgumentParser()
    main_parser.add_argument(
        '--experiment',
        type=str,
        default='conv_small',
        choices=BASELINE_EXPERIMENTS,
        help=f'Experiment choice, one of {BASELINE_EXPERIMENTS}'
    )
    
    pipeline = DMRIRSupervisedPipeline(main_parser)
    pipeline.init_pipeline("./configs/supervised_dmrir.yaml")

    train_loader, val_loader, test_loader = pipeline.prepare_data()
    
    model = MODELS[pipeline.config['experiment']]().to(pipeline.config['device'])
    with Logger(pipeline.config) as logger:
        trainer = pipeline.prepare_trainer(model, logger)
        pipeline.run(trainer, train_loader, val_loader, test_loader)
    
    