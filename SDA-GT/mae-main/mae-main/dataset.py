import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
from torchvision.transforms.functional import _interpolation_modes_from_int, InterpolationMode
from typing import Optional, Union
import random


class RandomResizedCrop(object):
    def __init__(
            self,
            size,
            scale=(0.08, 1.0),
            ratio=(3.0 / 4.0, 4.0 / 3.0),
            interpolation=InterpolationMode.BILINEAR,
            antialias: Optional[Union[str, bool]] = "warn",
    ):
        super().__init__()

        self.size = size, size

        if isinstance(interpolation, int):
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        # self.antialias = antialias
        self.scale = scale
        self.ratio = ratio

    def __call__(self, image, fuse, back, target=None):
        i, j, h, w = T.RandomResizedCrop.get_params(image, self.scale, self.ratio)
        # image = F.resized_crop(image, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)
        # fuse = F.resized_crop(fuse, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)
        # if target is not None:
        #     target = F.resized_crop(target, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)

        image = F.resized_crop(image, i, j, h, w, self.size, self.interpolation)
        fuse = F.resized_crop(fuse, i, j, h, w, self.size, self.interpolation)
        back = F.resized_crop(back, i, j, h, w, self.size, self.interpolation)
        if target is not None:
            target = F.resized_crop(target, i, j, h, w, self.size, self.interpolation)
        return image, fuse, back, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob = 0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, fuse, back, target=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            fuse = F.hflip(fuse)
            back = F.hflip(back)
            if target is not None:
                target = F.hflip(target)
        return image, fuse, back, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, fuse, back, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        fuse = F.normalize(fuse, mean=self.mean, std=self.std)
        back = F.normalize(back, mean=self.mean, std=self.std)
        return image, fuse, back, target


class ToTensor(object):
    def __call__(self, image, fuse, back, target):
        image = F.to_tensor(image)
        fuse = F.to_tensor(fuse)
        back = F.to_tensor(back)
        if target is not None:
            target = (torch.as_tensor(np.array(target), dtype=torch.int64) / 255).unsqueeze(0)  # 非0即1
        return image, fuse, back, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, fuse=None, back=None, mask=None):
        for t in self.transforms:
            image, fuse, back, mask = t(image, fuse, back, mask)
        return image, fuse, back, mask


class InfDataset(Dataset):
    def __init__(self,in_dir1,in_dir2, in_dir3, in_dir4, transform):
        super(InfDataset, self).__init__()

        self.in_dir1 = in_dir1
        self.in_dir2 = in_dir2
        self.in_dir3 = in_dir3
        self.in_dir4 = in_dir4
        self.transform = transform
        self.in_files1 = os.listdir(self.in_dir1)

    def __len__(self):
        return len(self.in_files1)

    def __getitem__(self, index):
        # input, mask, fusion
        in_file1 = self.in_files1[index]
        in_path1 = os.path.join(self.in_dir1, in_file1)  # input
        in_path2 = os.path.join(self.in_dir2, in_file1)  # fusion
        in_path3 = os.path.join(self.in_dir3, in_file1)  # mask
        in_path4 = os.path.join(self.in_dir4, in_file1)  # background

        input = Image.open(in_path1).convert('L')
        fuse = Image.open(in_path2).convert('L')
        mask = Image.open(in_path3).convert('L')
        back = Image.open(in_path4).convert('L')

        input, fuse, back, mask = self.transform(input, fuse, back, mask)

        return input, fuse, back, mask


class InfDataset_Test(Dataset):
    def __init__(self,in_dir1, in_dir2,transform):
        super(InfDataset_Test, self).__init__()

        self.in_dir1 = in_dir1
        self.in_dir2 = in_dir2
        self.transform = transform
        self.in_files1 = os.listdir(self.in_dir1)

    def __len__(self):
        return len(self.in_files1)

    def __getitem__(self, index):
        # input, mask
        in_file1 = self.in_files1[index]
        in_path1 = os.path.join(self.in_dir1, in_file1)  # input
        in_path2 = os.path.join(self.in_dir2, in_file1)  # mask

        input = Image.open(in_path1).convert('L')
        mask = Image.open(in_path2).convert('L')

        input = self.transform(input)
        mask = (torch.as_tensor(np.array(mask), dtype=torch.int64) / 255).unsqueeze(0)  # 非0即1

        return input, mask, in_file1