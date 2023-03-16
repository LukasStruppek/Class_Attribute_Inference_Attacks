import os
import random
from collections import Counter
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import PIL
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, Subset
from torchvision.datasets import CelebA, VisionDataset
from torchvision.datasets.utils import verify_str_arg


class CelebAIdentities(Dataset):

    def __init__(self,
                 train,
                 attribute_group,
                 split_seed=42,
                 transform=None,
                 target_transform=None,
                 root='data/celeba'):

        self.root = root

        if attribute_group.lower() == 'race':
            label_path = "labels/celeba_subset_identities/celeba400_race.csv"
        elif attribute_group.lower() == 'race_large':
            label_path = "labels/celeba_subset_identities/celeba1000_race.csv"
        elif attribute_group.lower() == 'gender':
            label_path = "labels/celeba_subset_identities/celeba500_gender.csv"
        elif attribute_group.lower() == 'gender_large':
            label_path = "labels/celeba_subset_identities/celeba1000_gender.csv"
        elif attribute_group.lower() == 'hair_color':
            label_path = "labels/celeba_subset_identities/celeba400_haircolor.csv"
        elif attribute_group.lower() == 'hair_color_large':
            label_path = "labels/celeba_subset_identities/celeba1000_haircolor.csv"
        elif attribute_group.lower() == 'eyeglasses':
            label_path = "labels/celeba_subset_identities/celeba200_eyeglasses.csv"
        elif attribute_group.lower() == 'eyeglasses_large':
            label_path = "labels/celeba_subset_identities/celeba1000_eyeglasses.csv"
        else:
            raise Exception(
                f'{attribute_group} is no valid attribute group. Select one of [race, gender, eyeglasses, hair_color, race_large, gender_large, hair_color_large, eyeglasses_large].'
            )

        labels = pd.read_csv(label_path, sep=',')

        # Select the corresponding samples for train and test split
        indices = list(range(len(labels.index)))
        np.random.seed(split_seed)
        np.random.shuffle(indices)
        training_set_size = int(0.9 * len(indices))
        train_idx = indices[:training_set_size]
        test_idx = indices[training_set_size:]

        # Assert that there are no overlapping datasets
        assert len(set.intersection(set(train_idx), set(test_idx))) == 0

        # Set transformations
        self.transform = transform
        self.target_transform = target_transform

        # Split dataset
        if train:
            self.filenames = np.array(labels['file'])[train_idx]
            self.targets = np.array(labels['idx'])[train_idx]
            self.name = f'CelebA_{attribute_group}_train'
        else:
            self.filenames = np.array(labels['file'])[test_idx]
            self.targets = np.array(labels['idx'])[test_idx]
            self.name = f'CelebA_{attribute_group}_test'

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        file_path = os.path.join(self.root, "img_align_celeba",
                                 self.filenames[index])
        if os.path.exists(file_path) == False:
            file_path = file_path.replace('.jpg', '.png')
        X = PIL.Image.open(file_path)

        target = self.targets[index]

        if self.transform is not None:
            X = self.transform(X)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return X, target


class CelebAAttributes(Dataset):

    def __init__(self,
                 train,
                 split_seed=42,
                 transform=None,
                 attribute_group=None,
                 root='data/celeba',
                 download: bool = False,
                 balanced_sampling=False,
                 top_1000_identities=False):
        # Load default CelebA dataset
        celeba = CustomCelebA(root=root, split='all', target_type="attr")
        celeba.targets = celeba.attr
        self.top_1000_identities = top_1000_identities
        attributes = celeba.attr

        self.attribute_list = [
            '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
            'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
            'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
            'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
            'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
            'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
            'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
            'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
            'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
            'Wearing_Necklace', 'Wearing_Necktie', 'Young'
        ]

        self.attribute_to_idx = {
            attr: idx
            for idx, attr in enumerate(self.attribute_list)
        }
        self.idx_to_attribute = {
            idx: attr
            for idx, attr in enumerate(self.attribute_list)
        }

        # Unselect the 1,000 most frequent celebrities from the dataset
        identity_targets = np.array([t.item() for t in celeba.identity])
        ordered_dict = dict(
            sorted(Counter(identity_targets).items(),
                   key=lambda item: item[1],
                   reverse=True))

        if top_1000_identities:
            sorted_identities = list(ordered_dict.keys())[:1000]
        else:
            sorted_identities = list(ordered_dict.keys())[1000:]

        # Select the corresponding samples for train and test split
        indices = np.where(np.isin(identity_targets, sorted_identities))[0]

        # Set transformations
        self.transform = transform

        if attribute_group == 'gender':
            self.target_transform = lambda t: t[self.attribute_to_idx['Male']]
        elif attribute_group == 'eyeglasses':
            self.target_transform = lambda t: t[self.attribute_to_idx['Male']]
        elif attribute_group in ['hair_color', 'hair_color_no_bald']:
            if attribute_group == 'hair_color':
                self.target_transform = lambda t: [
                    t[self.attribute_to_idx[hair_color]] for hair_color in [
                        'Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair',
                        'Gray_Hair'
                    ]
                ]
            elif attribute_group == 'hair_color_no_bald':
                self.target_transform = lambda t: [
                    t[self.attribute_to_idx[hair_color]] for hair_color in
                    ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
                ]

            # Select the corresponding samples which only a single hair color is defined
            hair_color_list = np.where(
                np.sum(np.array([self.target_transform(t)
                                 for t in attributes]),
                       axis=1) == 1)[0]
            indices = np.intersect1d(hair_color_list, indices)
            if attribute_group == 'hair_color':
                self.target_transform = lambda t: [
                    t[self.attribute_to_idx[hair_color]] for hair_color in [
                        'Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair',
                        'Gray_Hair'
                    ]
                ].index(1)
            elif attribute_group == 'hair_color_no_bald':
                self.target_transform = lambda t: [
                    t[self.attribute_to_idx[hair_color]] for hair_color in
                    ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
                ].index(1)
        else:
            self.target_transform = lambda t: t

        np.random.seed(split_seed)
        np.random.shuffle(indices)
        training_set_size = int(0.9 * len(indices))
        train_idx = indices[:training_set_size]
        test_idx = indices[training_set_size:]

        if balanced_sampling:
            train_targets = np.array(
                [self.target_transform(t) for t in attributes[train_idx]])
            values, counts = np.unique(train_targets, return_counts=True)
            samples_per_class = len(train_targets) // len(values)
            train_idx = np.array(train_idx)
            train_idx_new = []
            for value, count in zip(values, counts):
                if count > samples_per_class:
                    indices_value = train_idx[np.argwhere(
                        train_targets == value).flatten()
                                              [:samples_per_class].tolist()]
                else:
                    indices_value = train_idx[random.choices(
                        np.argwhere(train_targets == value).flatten(),
                        k=samples_per_class)]
                train_idx_new += indices_value.tolist()
            train_idx = train_idx_new

        # Assert that there are no overlapping datasets
        assert len(set.intersection(set(train_idx), set(test_idx))) == 0

        # Split dataset
        if train:
            self.dataset = Subset(celeba, train_idx)
            train_targets = np.array(attributes)[train_idx]
            self.targets = [self.target_transform(t) for t in train_targets]
            self.name = 'CelebAAttributes_train'
        else:
            self.dataset = Subset(celeba, test_idx)
            test_targets = np.array(attributes)[test_idx]
            self.targets = [self.target_transform(t) for t in test_targets]
            self.name = 'CelebAAttributes_test'

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        im, _ = self.dataset[idx]
        if self.transform:
            return self.transform(im), self.targets[idx]
        else:
            return im, self.targets[idx]


class CustomCelebA(VisionDataset):
    """ 
    Modified CelebA dataset to adapt for custom cropped images.
    """

    def __init__(
        self,
        root: str,
        split: str = "all",
        target_type: Union[List[str], str] = "identity",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super(CustomCelebA, self).__init__(root,
                                           transform=transform,
                                           target_transform=target_transform)
        self.split = split
        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        if not self.target_type and self.target_transform is not None:
            raise RuntimeError(
                'target_transform is specified but target_type is empty')

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[verify_str_arg(split.lower(), "split",
                                          ("train", "valid", "test", "all"))]

        fn = partial(os.path.join, self.root)
        splits = pd.read_csv(fn("list_eval_partition.txt"),
                             delim_whitespace=True,
                             header=None,
                             index_col=0)
        identity = pd.read_csv(fn("identity_CelebA.txt"),
                               delim_whitespace=True,
                               header=None,
                               index_col=0)
        bbox = pd.read_csv(fn("list_bbox_celeba.txt"),
                           delim_whitespace=True,
                           header=1,
                           index_col=0)
        landmarks_align = pd.read_csv(fn("list_landmarks_align_celeba.txt"),
                                      delim_whitespace=True,
                                      header=1)
        attr = pd.read_csv(fn("list_attr_celeba.txt"),
                           delim_whitespace=True,
                           header=1)
        mask = slice(None) if split_ is None else (splits[1] == split_)

        self.filename = splits[mask].index.values
        self.identity = torch.as_tensor(identity[mask].values)
        self.bbox = torch.as_tensor(bbox[mask].values)
        self.landmarks_align = torch.as_tensor(landmarks_align[mask].values)
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = torch.div(
            self.attr + 1, 2,
            rounding_mode='floor')  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        file_path = os.path.join(self.root, "img_align_celeba",
                                 self.filename[index])
        if os.path.exists(file_path) == False:
            file_path = file_path.replace('.jpg', '.png')
        X = PIL.Image.open(file_path)

        target: Any = []
        for t in self.target_type:
            if t == "attr":
                target.append(self.attr[index, :])
            elif t == "identity":
                target.append(self.identity[index, 0])
            elif t == "bbox":
                target.append(self.bbox[index, :])
            elif t == "landmarks":
                target.append(self.landmarks_align[index, :])
            else:
                raise ValueError(
                    "Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)
