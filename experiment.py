from abc import ABCMeta, abstractmethod
from argparse import ArgumentParser
from enum import Enum


class ModelTypes(Enum):
    LSAE='lsae'
    AE='ae'


class Classifiers(Enum):
    SVM='svm'
    LINEAR='linear'


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
            '--classifier',
            type=str,
            default=Classifiers.SVM.value,
            choices=[c.value for c in Classifiers],
            help=f"Classifier choice for breast cancer detection"
        )
        
        self.main_parser.add_argument(
            '--checkpoint',
            type=str,
            required=False,
            help='A path to the .pth file of the trained SAE model'
        )
        
        self.main_parser.add_argument(
            '--test-only', 
            action="store_true",
            help="Run the breats cancer classification only, requires a trained encoder (specified in --checkpoint argument)"
        )
        
    
    @abstractmethod
    def run(self):
        pass
    
    def test(self):
        pass
        