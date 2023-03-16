import argparse
import math
import pickle
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms.functional as F
from rtpt import RTPT
from sklearn import metrics

import wandb
from utils.wandb import load_model


def create_image(w,
                 generator,
                 crop_size=None,
                 resize=None,
                 batch_size=20,
                 device='cuda:0'):
    with torch.no_grad():
        if w.shape[1] == 1:
            w_expanded = torch.repeat_interleave(w,
                                                 repeats=generator.num_ws,
                                                 dim=1)
        else:
            w_expanded = w

        w_expanded = w_expanded.to(device)
        imgs = []
        for i in range(math.ceil(w_expanded.shape[0] / batch_size)):
            w_batch = w_expanded[i * batch_size:(i + 1) * batch_size]
            imgs_generated = generator(w_batch,
                                       noise_mode='const',
                                       force_fp32=True)
            imgs.append(imgs_generated.cpu())

        imgs = torch.cat(imgs, dim=0)
        if crop_size is not None:
            imgs = F.center_crop(imgs, (crop_size, crop_size))
        if resize is not None:
            imgs = F.resize(imgs, resize)
        return imgs


def load_generator(filepath):
    with open(filepath, 'rb') as f:
        sys.path.insert(0, 'stylegan2-ada-pytorch')
        G = pickle.load(f)['G_ema'].cuda()
    return G


def load_labels(path, attribute_class):
    df = pd.read_csv(path)
    df = df.fillna(np.nan).replace([np.nan], [None])
    label_dict = df[attribute_class.lower()].to_dict()
    return label_dict


def get_label_mapping(attribute):
    if attribute == 'eyeglasses':
        return {0: 'no_eyeglasses', 1: 'eyeglasses'}
    if attribute == 'gender':
        return {0: 'female', 1: 'male'}
    if attribute == 'hair_color':
        return {
            0: 'black_hair',
            1: 'blond_hair',
            2: 'brown_hair',
            3: 'gray_hair'
        }
    if attribute == 'race':
        return {0: 'white', 1: 'black', 2: 'asian', 3: 'indian'}


def create_model(path):
    if 'fairface' in path:
        model = torchvision.models.resnet34()
        model.fc = torch.nn.Linear(model.fc.in_features, 18)
        model.load_state_dict(torch.load(path))
    else:
        model = load_model(path)
    model.eval()
    model.cuda()
    return model


def main():
    torch.manual_seed(0)
    parser = create_parser()
    args = parser.parse_args()

    # Load filter model
    filter_model = create_model(args.filter_model).eval().cuda()

    # Load PPA results
    w_opt = torch.load(
        wandb.restore(
            f'results/optimized_w_selected_{args.ppa_path.split("/")[-1]}.pt',
            run_path=args.ppa_path).name)

    # Start RTPT
    rtpt = RTPT(args.user, 'PPA Eval', int(len(w_opt) / 25))
    rtpt.start()

    # Load StyleGAN generator
    G = load_generator(args.stylegan).eval().cuda()

    # Load attribute labels
    attr_true = load_labels(args.labels, args.attribute)
    ground_truth = OrderedDict(sorted(attr_true.items()))
    ground_truth = list(ground_truth.values())
    labels_pred = []
    labels = get_label_mapping(args.attribute)

    # Initialize WandB Logging
    wandb.init(project='model_inversion_attacks_CAIA_Results',
               save_code=True,
               name=args.attribute)

    # Perform attribute inference
    for idx in range(int(len(w_opt) / args.num_samples)):
        with torch.no_grad():
            w_batch = w_opt[idx * args.num_samples:(idx + 1) *
                            args.num_samples].cuda()
            w_batch = torch.repeat_interleave(w_batch, repeats=G.num_ws, dim=1)
            imgs = G.synthesis(w_batch)
            imgs = F.resize(imgs, (224, 224))
            outputs = filter_model(imgs)[:, :len(labels)]
            pred = torch.argmax(outputs, dim=1).flatten()
            attr_pred = torch.mode(pred).values.cpu().item()
            attr_pred = labels[attr_pred]
            print(idx, attr_pred, ground_truth[idx])
            labels_pred.append(attr_pred)
            rtpt.step()

    metric_dict = metrics.classification_report(ground_truth,
                                                labels_pred,
                                                digits=4,
                                                output_dict=True)

    print(metric_dict)
    wandb.run.summary["num_targets"] = len(attr_true)
    wandb.run.summary.update(metric_dict)
    wandb.sklearn.plot_confusion_matrix(ground_truth, labels_pred)
    wandb.finish()


def create_parser():
    parser = argparse.ArgumentParser(
        description='Performing model inversion attack')
    parser.add_argument('-f',
                        '--filter_model',
                        default=None,
                        type=str,
                        dest="filter_model",
                        help='Define filter model WandB runpath')
    parser.add_argument('-p',
                        '--ppa',
                        default=None,
                        type=str,
                        dest="ppa_path",
                        help='Define PPA WandB runpath')
    parser.add_argument('-l',
                        '--labels',
                        default=None,
                        type=str,
                        dest="labels",
                        help='Define path to class attribute labels')
    parser.add_argument('-a',
                        '--attribute',
                        default=None,
                        type=str,
                        dest="attribute",
                        help='Define the target attribute')
    parser.add_argument('-u',
                        '--user',
                        default='XX',
                        type=str,
                        dest="user",
                        help='Define RTPT User (Default: XX)')
    parser.add_argument(
        '-s',
        '--stylegan',
        default='stylegan2-ada-pytorch/ffhq.pkl',
        type=str,
        dest="stylegan",
        help=
        'Define StyleGAN2 model path (Default: stylegan2-ada-pytorch/ffhq.pkl)'
    )
    parser.add_argument(
        '-n',
        '--num_samples',
        default=25,
        type=int,
        dest="num_samples",
        help='Number of PPA samples per target identitiy (Default: 25)')

    return parser


if __name__ == '__main__':
    main()
