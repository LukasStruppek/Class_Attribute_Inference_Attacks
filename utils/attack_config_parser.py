from copy import copy
from typing import List

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as T
import yaml
from datasets.single_image_folder import SingleImageFolder
from rtpt import RTPT

import wandb
from utils.wandb import load_model


class AttackConfigParser:

    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        self._config = config

    def has_filter_model(self):
        if 'attribute_classifier' in self._config:
            return True
        else:
            return False

    def create_model(self, mode, device):
        if 'wandb_run' in self._config[mode]:
            model = load_model(self._config[mode]['wandb_run'])
        elif 'fairface' in self._config[mode]:
            model = torchvision.models.resnet34()
            model.fc = torch.nn.Linear(model.fc.in_features, 18)
            model.load_state_dict(torch.load(self._config[mode]['fairface']))
            model.device = device
        else:
            print(
                'No target model stated in the config file. No filtering applied!'
            )
            return None

        model.eval()
        model.to(device)
        return model

    def create_transformations(self, mode):
        """
        mode: 'attribute_classifier' or 'target_model'
        """

        transformation_list = [T.ToTensor()]
        if mode == 'attribute_classifier':
            transformations = self.attribute_classifier['transformations']
        elif mode == 'target_model':
            transformations = self.target_model['transformations']
        elif mode == 'attack_transformations':
            if 'transformations' in self.attack:
                transformations = self.attack['transformations']
            else:
                return None
        else:
            raise Exception(
                f'{mode} is no valid transformation mode. Please choose "target_model", "attack_transformations" or "attribute_classifier"'
            )

        transformation_list = []
        if mode != 'attack_transformations':
            transformation_list.append(T.ToTensor())
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

    def load_candidate_datasets(self, transformations=None):
        dataset_list = []
        for idx in self.candidates:
            label = self.candidates[idx][0]
            dataset = SingleImageFolder(self.candidates[idx][1],
                                        transform=transformations,
                                        target=idx)
            dataset_list.append((label, dataset))
        assert all(
            len(dataset) == len(dataset_list[0][1])
            for label, dataset in dataset_list)
        return dataset_list

    def load_labels(self):
        df = pd.read_csv(self.attack['labels'])
        df = df.fillna(np.nan).replace([np.nan], [None])
        label_dict = df[self.attack['attribute_class'].lower()].to_dict()
        return label_dict

    def get_target_dataset(self):
        try:
            api = wandb.Api(timeout=60)
            run = api.run(self._config['wandb_target_run'])
            return run.config['Dataset'].strip().lower()
        except:
            return self._config['dataset']

    def create_wandb_config(self):
        config = {**self.attack}
        config['target_model'] = self.target_model
        config['attribute_classifier'] = self.attribute_classifier
        return config

    def get_target_classes(self, num_classes=None):
        targets = self._config['attack']['targets']
        if isinstance(targets, int):
            return 1
        elif isinstance(targets, list):
            return targets
        elif targets == 'all' and num_classes is not None:
            return [i for i in range(num_classes)]
        else:
            raise Exception(f'{targets} is no valid target specification.')

    def create_rtpt(self, max_iterations):
        rtpt_config = self._config['rtpt']
        rtpt = RTPT(name_initials=rtpt_config['name_initials'],
                    experiment_name=rtpt_config['experiment_name'],
                    max_iterations=max_iterations)
        return rtpt

    @property
    def candidates(self):
        return self._config['candidates']

    @property
    def attribute_classifier(self):
        if 'attribute_classifier' in self._config:
            return self._config['attribute_classifier']
        else:
            return None

    @property
    def target_model(self):
        return self._config['target_model']

    @property
    def wandb_target_run(self):
        return self._config['wandb_target_run']

    @property
    def logging(self):
        return self._config['wandb']['enable_logging']

    @property
    def wandb_init_args(self):
        return self._config['wandb']['wandb_init_args']

    @property
    def attack(self):
        return self._config['attack']

    @property
    def candidates(self):
        return self._config['candidates']

    @property
    def wandb(self):
        return self._config['wandb']

    @property
    def seed(self):
        return self._config['seed']

    @property
    def batch_size(self):
        return self._config['batch_size']

    @property
    def dataset(self):
        return self._config['dataset']

    @property
    def log_progress(self):
        if 'log_progress' in self._config['attack']:
            return self._config['attack']['log_progress']
        else:
            return True

    @property
    def filter_threshold(self):
        if 'filter_threshold' in self.attack:
            return self.attack['filter_threshold']
        else:
            return 0.0

    @property
    def num_variations(self):
        if 'num_variations' in self.attack:
            return self.attack['num_variations']
        else:
            return 1.0

    @property
    def majority_vote(self):
        if 'majority_vote' in self.attack:
            return self.attack['majority_vote']
        else:
            return True

    @property
    def quantile(self):
        if 'quantile' in self.attack:
            return self.attack['quantile']
        else:
            return 1.0

    @property
    def lower_index_bound(self):
        if 'attack_sample_range' in self.attack:
            return self.attack['attack_sample_range'][0]
        else:
            return 0

    @property
    def upper_index_bound(self):
        if 'attack_sample_range' in self.attack:
            return self.attack['attack_sample_range'][1]
        else:
            return 1000000
