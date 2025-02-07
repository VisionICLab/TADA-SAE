from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser

class AbstractExperiment(metaclass=ABCMeta):
    def __init__(self, experiment_choices):
        self.main_parser = ArgumentParser()

        self.main_parser.add_argument(
            '--experiment',
            type=str,
            default='tada_sae',
            choices=experiment_choices,
            help=f'Experiment choice, one of {experiment_choices}'
        )
        
        self.main_parser.add_argument(
            '--checkpoint',
            type=str,
            required=False,
            help='A path to the .pth file of the trained SAE model'
        )
        
        self.main_parser.add_argument('--test_only', action="store_true")
        

    @abstractmethod
    def run(self):
        pass
    
    def test(self):
        pass
        