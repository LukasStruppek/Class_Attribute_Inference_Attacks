from typing import List

import torch
import torchvision
import torchvision.transforms as T
import yaml
from rtpt.rtpt import RTPT
from torchvision.datasets import *

from utils.wandb import load_model


class SynthesizingConfigParser:

    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        self._config = config

    def create_filter_transformations(self):
        dataset_config = self._config['filter_model']
        transformation_list = []
        transformations = dataset_config['transformations']
        if transformations != None:
            for transform, args in transformations.items():
                if not hasattr(T, transform):
                    raise Exception(
                        f'{transform} is no valid transformation. Please write the type exactly as the Torchvision class'
                    )
                else:
                    transformation_class = getattr(T, transform)
                    transformation_list.append(transformation_class(**args))
        data_transformation = T.Compose(transformation_list)
        return data_transformation

    def create_rtpt(self, max_iterations=1):
        rtpt_config = self._config['rtpt']
        rtpt = RTPT(name_initials=rtpt_config['name_initials'],
                    experiment_name=rtpt_config['experiment_name'],
                    max_iterations=max_iterations)
        return rtpt

    def load_filter_model(self):
        if 'wandb_run' in self._config['filter_model']:
            filter_model = load_model(
                self._config['filter_model']['wandb_run'])
        elif 'fairface' in self._config['filter_model']:
            filter_model = torchvision.models.resnet34()
            filter_model.fc = torch.nn.Linear(filter_model.fc.in_features, 18)
            filter_model.load_state_dict(
                torch.load(self._config['filter_model']['fairface']))

        filter_model.eval()
        return filter_model

    @property
    def image_latents(self):
        return self._config['generation']['image_latents']

    @property
    def ldm_model_path(self):
        return self._config['stable_diffusion']['model_path']

    @property
    def attributes(self):
        return self._config['generation']['attributes']

    @property
    def attribute_class(self):
        return self._config['generation']['attribute_class']

    @property
    def cross_replace_steps(self):
        return self._config['generation']['cross_replace_steps']

    @property
    def self_replace_steps(self):
        return self._config['generation']['self_replace_steps']

    @property
    def threshold(self):
        return self._config['filter_model']['threshold']

    @property
    def num_samples(self):
        return self._config['generation']['num_samples']

    @property
    def output_folder(self):
        return self._config['generation']['output_folder']
