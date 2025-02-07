from argparse import ArgumentParser
from experiment import AbstractExperiment


class TADASAEAblationExperiments(AbstractExperiment):
    def __init__(self):
        super().__init__(['ae_full_im', 'ae_left_right', 'lsae_left_right', 'lsae_full_im'])
        

