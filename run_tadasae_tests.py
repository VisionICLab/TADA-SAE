from argparse import ArgumentParser

DATASETS=['dmrir', 'orthopot']
EXPERIMENTS=['ae_full_im', 'ae_left_right', 'lsae_left_right', 'lsae_full_im', 'tada_sae']


class TADASAEAblationExperiments:
    def __init__(self):
        main_parser = ArgumentParser()

        main_parser.add_argument(
            '--dataset',
            type=str,
            default='dmrir',
            choices=DATASETS,
            help=f'Dataset choice, one of {DATASETS}'
        )

        main_parser.add_argument(
            '--experiment',
            type=str,
            default='tada_sae',
            choices=DATASETS,
            help=f'Experiment choice, one of {EXPERIMENTS}'
        )
        args = vars(main_parser.parse_args())
        self.dataset = args['dataset']
        self.experiment = args['experiment']

