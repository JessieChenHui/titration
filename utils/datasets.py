#!/usr/bin/python
# -*- coding: UTF-8 -*-
# create date: 2025/2/1
# __author__: 'Alex Lu'
import os
import torch

from torch.utils.data import Dataset
import pandas as pd
from PIL import Image


class ImageFileDataset(Dataset):
    def __init__(self, ds, root_dir, transform=None):
        """
        Args:
            ds (pd.DataFrame): 包含标签与图像路径信息。
            root_dir (string): 图像文件的根目录。
            transform (callable, optional): 可选的图像预处理函数。
        """
        self.ds = ds
        self.root_dir = root_dir
        self.transform = transform

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.ds.iloc[idx, :]
        image_file = f'{row.iloc[1]}_{row.iloc[2]}.png'
        image_path = os.path.join(self.root_dir, row.iloc[1], image_file)
        image = Image.open(image_path).convert('RGB')
        label = int(row.iloc[0])

        if self.transform:
            image = self.transform(image)

        return image, label


class ImagesDataset(Dataset):
    def __init__(self, images, transform=None):
        """
        Args:
            images (list): 图像文件的根目录。
            transform (callable, optional): 可选的图像预处理函数。
        """
        self.images = images
        self.transform = transform

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        # print(type(image))
        if self.transform:
            image = self.transform(image)

        return image, 1


class ImageDataset(Dataset):
    def __init__(self, ds, root_dir, transform=None):
        """
        Args:
            ds (pd.DataFrame): 包含标签与图像路径信息。
            root_dir (string): 图像文件的根目录。
            transform (callable, optional): 可选的图像预处理函数。
        """
        self.ds = ds
        self.root_dir = root_dir
        self.transform = transform

    def set_transform(self, transform):
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.ds.iloc[idx, :]
        image_file = f'{row.iloc[1]}_{row.iloc[2]}.png'
        image_path = os.path.join(self.root_dir, row.iloc[1], image_file)
        image = Image.open(image_path).convert('RGB')
        label = int(row.iloc[0])

        if self.transform:
            image = self.transform(image)

        return image, label