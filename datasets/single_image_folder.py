import torch
import os
from PIL import Image


class SingleImageFolder(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, target=None):
        self.root = root
        self.transform = transform
        self.imgs = sorted(os.listdir(root))
        if target is not None:
            self.targets = [target for i in range(len(self.imgs))]
        else:
            self.targets = None

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.targets is not None:
            return img, self.targets[idx]
        else:
            return img
