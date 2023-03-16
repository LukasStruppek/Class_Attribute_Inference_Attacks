import argparse
import random
from collections import OrderedDict
from copy import deepcopy
from sys import exit

import numpy as np
import torch
import wandb
from sklearn import metrics
from torch.utils.data import DataLoader

from datasets.custom_subset import Subset
from utils.attack_config_parser import AttackConfigParser
from utils.wandb import *


def main():
    ####################################
    #        Attack Preparation        #
    ####################################

    # Set devices
    torch.set_num_threads(24)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define and parse attack arguments
    parser = create_parser()
    config, args = parse_arguments(parser)

    # Set seeds
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Load basic attack parameters
    batch_size = config.attack['batch_size']
    target_model = config.create_model(mode='target_model', device=device)
    target_classes = config.get_target_classes(target_model.num_classes)

    # Initialize wandb logging
    if config.logging:
        wandb_run = init_wandb_logging(target_model.name,
                                       target_model.architecture, config, args)

    # Initialize RTPT
    rtpt = config.create_rtpt(max_iterations=len(target_classes))
    rtpt.start()

    torch.set_num_threads(8)

    ####################################
    #           Filter Images          #
    ####################################

    if config.has_filter_model():
        attribute_classifier = config.create_model(mode='attribute_classifier',
                                                   device=device)
        attribute_classifier_transforms = config.create_transformations(
            'attribute_classifier')
        candidate_sets = config.load_candidate_datasets(
            attribute_classifier_transforms)
        num_candidates = len(candidate_sets[0][1])
        print(
            f'Start filtering datasets. Total number of candidates: {num_candidates}. Confidence threshold: {config.filter_threshold}.'
        )
        filtered_indices = filter(attribute_classifier,
                                  candidate_sets,
                                  batch_size=batch_size,
                                  threshold=config.filter_threshold)
    else:
        candidate_sets = config.load_candidate_datasets(None)
        num_candidates = len(candidate_sets[0][1])
        filtered_indices = torch.ones(num_candidates).nonzero().cpu()

    filtered_indices = filtered_indices[config.lower_index_bound:config.
                                        upper_index_bound]
    num_filtered = len(filtered_indices)
    print(
        f'Finished filtering datasets. Number of retained images: {num_filtered} ({100*num_filtered/num_candidates:.2f}%)'
    )
    if num_filtered == 0:
        print(
            'No retained images. Stopping the attack. Please check the defined attribute indices in the config file for possible mistakes.'
        )
        exit(0)

    if config.logging:
        wandb.run.summary["candidates"] = num_candidates
        wandb.run.summary["num_filtered"] = num_filtered
        wandb.run.summary["filter_ratio"] = num_filtered / num_candidates

    target_model_transforms = config.create_transformations('target_model')
    filtered_subsets = create_filtered_subsets(
        candidate_sets, filtered_indices, transforms=target_model_transforms)

    ####################################
    #         Attack Iteration         #
    ####################################

    attr_true = config.load_labels()

    attack_transforms = config.create_transformations('attack_transformations')

    print(
        f'Start attack. Total number of targets: {len(target_classes)}. Applying majority vote: {config.majority_vote}.'
    )
    if attack_transforms is not None:
        print(f'Attack Transformations applied {config.num_variations} times:')
        for transform in attack_transforms.transforms:
            print(f'\t {transform}')

    attr_pred = predict_attribute(filtered_subsets, target_model, device,
                                  batch_size, config.majority_vote,
                                  attack_transforms, config.num_variations)

    ####################################
    #         Attack Metrics           #
    ####################################
    ground_truth = OrderedDict(sorted(attr_true.items()))
    ground_truth = list(ground_truth.values())
    assert len(ground_truth) == len(attr_pred)

    metric_dict = metrics.classification_report(ground_truth,
                                                attr_pred,
                                                digits=4,
                                                output_dict=True)

    ####################################
    #          Finish Logging          #
    ####################################
    if config.logging:
        wandb.run.summary["num_targets"] = len(attr_true)
        wandb.run.summary.update(metric_dict)
        wandb.sklearn.plot_confusion_matrix(list(attr_true.values()),
                                            list(attr_pred))
        wandb.finish()
    else:
        print('Attack finished')
        print(metric_dict)


def create_parser():
    parser = argparse.ArgumentParser(
        description='Performing model inversion attack')
    parser.add_argument('-c',
                        '--config',
                        default=None,
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    parser.add_argument('--no_rtpt',
                        action='store_false',
                        dest="rtpt",
                        help='Disable RTPT')
    return parser


def parse_arguments(parser):
    args = parser.parse_args()

    if not args.config:
        print(
            "Configuration file is missing. Please check the provided path. Execution is stopped."
        )
        exit()

    # Load attack config
    config = AttackConfigParser(args.config)

    return config, args


def filter(model,
           dataset_list,
           batch_size=32,
           idx_mapping=None,
           threshold=0.0):
    correct_predictions = []

    for target, dataset in enumerate(dataset_list):
        label, dataset = dataset
        dataloader = DataLoader(dataset,
                                batch_size,
                                shuffle=False,
                                drop_last=False)
        predictions = []
        for batch in dataloader:
            with torch.no_grad():
                if isinstance(batch, list):
                    batch = batch[0]
                batch = batch.to(model.device)
                output = model(batch)[:, :len(dataset_list)].softmax(dim=1)
                pred = torch.argmax(output, dim=1).unsqueeze(1)
                pred[torch.max(output, dim=1).values < threshold] = -1
                predictions.append(pred)
        predictions = torch.cat(predictions, dim=0).cpu()
        if idx_mapping:
            target = idx_mapping[target]
        correct_predictions.append(predictions == target)
    correct_predictions = torch.cat(correct_predictions, dim=1)
    corrects = correct_predictions.sum(dim=1) == len(dataset_list)
    return corrects.nonzero().flatten().cpu()


def create_filtered_subsets(dataset_list, indices, transforms):
    filtered_subsets = {}
    for label, dataset in dataset_list:
        dataset.transform = transforms
        subset = Subset(dataset, indices)
        filtered_subsets[label] = subset
    return filtered_subsets


def predict_attribute(dataset_list,
                      target_model,
                      device,
                      batch_size=64,
                      majority_vote=True,
                      attack_transforms=None,
                      num_iterations=1):

    labels = list(dataset_list.keys())

    dataloader_list = [
        DataLoader(dataset_list[label],
                   batch_size,
                   shuffle=False,
                   drop_last=False) for label in labels
    ]

    if attack_transforms is None:
        num_iterations = 1

    final_list = []

    with torch.no_grad():
        for idx, dataloader in enumerate(dataloader_list):
            logit_list = []

            for batch in dataloader:
                imgs = batch[0].to(device)
                for i in range(num_iterations):
                    if attack_transforms is not None:
                        torch.manual_seed(i)
                        imgs = attack_transforms(imgs)

                    target_outputs = target_model(imgs)
                    logit_list.append(target_outputs)

            logits = torch.cat(logit_list, dim=0)
            final_list.append(logits.unsqueeze(-1))

        logits_cat = torch.cat(final_list, dim=-1).permute(1, 0, 2)
        values, indices = torch.topk(logits_cat, k=2, dim=-1)
        advantage = values[:, :, 0] - values[:, :, 1]
        indices = indices[:, :, 0]
        advantage_sum = torch.zeros((logits_cat.shape[0], logits_cat.shape[2]),
                                    device=device)
        votes = torch.zeros((logits_cat.shape[0], logits_cat.shape[2]),
                            device=device)
        for label in range(logits_cat.shape[2]):
            mask = (indices == label)
            advantage_masked = advantage * mask
            votes[:, label] += mask.sum(dim=1)
            advantage_sum[:, label] += advantage_masked.sum(dim=1)
        advantage_prediction = torch.argmax(advantage_sum, dim=1)
        majority_vote_prediction = torch.argmax(votes, dim=1)

        if majority_vote:
            predictions = majority_vote_prediction.cpu().numpy()
        else:
            predictions = advantage_prediction.cpu().numpy()

        # map idx to attribute values
        labels_pred = []
        for p in predictions:
            labels_pred.append(labels[p].lower())
        return labels_pred


if __name__ == '__main__':
    main()
