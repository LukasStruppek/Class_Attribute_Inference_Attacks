import argparse
import csv

import torch
import torchvision.transforms as T
from captum.attr import IntegratedGradients
from rtpt import RTPT
from torchvision.transforms.functional import resize
from tqdm import tqdm

from datasets.single_image_folder import SingleImageFolder
from utils.unet import Unet
from utils.wandb import load_model


def main():
    torch.manual_seed(0)
    parser = create_parser()
    args = parser.parse_args()

    # Load segmentation model
    unet = Unet()
    unet.load_state_dict(torch.load(args.segmentation_model))
    unet = unet.eval().cuda()

    # Define segmentation transforms
    transforms = T.Compose(
        [T.Resize((512, 512)),
         T.ToTensor(),
         T.Normalize(0.5, 0.5)])

    # Load classifier
    model = load_model(args.model_path, replace=False)

    # Setup Integrated Gradients
    integrated_gradients = IntegratedGradients(model)

    # Define attributes to measure the relative attribution for
    # See https://github.com/switchablenorms/CelebAMask-HQ/tree/master/face_parsing for textual labels
    attributes = {
        'eyeglasses': (3, ),
        'eyes': (4, 5),
        'hair': (13, ),
        'mouth': (10, 11, 12),
        'hat': (14, ),
        'cloth': (18, ),
        'skin': (1, ),
    }

    rel_attr_dict = {attr: [] for attr in attributes.keys()}
    dataset = SingleImageFolder(args.image_folder, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True)

    # Start RTPT
    rtpt = RTPT(args.user, 'Relative Attribution', model.num_classes)
    rtpt.start()

    for target in tqdm(range(model.num_classes)):
        for batch in dataloader:
            with torch.no_grad():
                batch = batch.cuda()
                output = unet(batch)
                output = resize(output,
                                (args.image_size, args.image_size)).permute(
                                    (0, 2, 3, 1))
                prediction = torch.argmax(output, dim=-1)

                attributions_ig = integrated_gradients.attribute(
                    resize(batch, (args.image_size, args.image_size)).cuda(),
                    target=target,
                    n_steps=50,
                    internal_batch_size=args.batch_size * 2).abs().cpu()

                for attr, values in attributes.items():

                    mask = prediction.cpu().apply_(lambda x: x in values).bool(
                    ).float().unsqueeze(1).repeat_interleave(3, dim=1)

                    attributions_masked = attributions_ig * mask

                    rel_attr = attributions_masked.sum(
                        dim=[1, 2, 3]) / attributions_ig.sum(dim=[1, 2, 3])

                    rel_attr_dict[attr].append(rel_attr.cpu())

        rtpt.step()

    for attr in attributes.keys():
        rel_attr = torch.cat(rel_attr_dict[attr], dim=0).mean().cpu().item()
        print(attr, rel_attr)
        with open(args.output_file, 'a', newline='') as csvfile:
            fieldnames = ['attribute', 'value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'attribute': attr, 'value': rel_attr})


def create_parser():
    parser = argparse.ArgumentParser(
        description='Performing model inversion attack')
    parser.add_argument('-c',
                        '--classifier_runpath',
                        default=None,
                        type=str,
                        dest="model_path",
                        help='Define classifier WandB runpath')
    parser.add_argument('-s',
                        '--segmentation_model',
                        default=None,
                        type=str,
                        dest="segmentation_model",
                        help='Define segmentation model path')
    parser.add_argument('-f',
                        '--image_folder',
                        default=None,
                        type=str,
                        dest="image_folder",
                        help='Define the image folder')
    parser.add_argument('-o',
                        '--output_file',
                        default='attribution.csv',
                        type=str,
                        dest="output_file",
                        help='Define output file (Default: attribution.csv)')
    parser.add_argument('-b',
                        '--batch_size',
                        default=10,
                        type=int,
                        dest="batch_size",
                        help='Batch size (Default: 10)')
    parser.add_argument('-i',
                        '--image_size',
                        default=224,
                        type=int,
                        dest="image_size",
                        help='Image Size (Default: 10)')
    parser.add_argument('-u',
                        '--user',
                        default='XX',
                        type=str,
                        dest="user",
                        help='Define RTPT User (Default: XX)')

    return parser


if __name__ == '__main__':
    main()
