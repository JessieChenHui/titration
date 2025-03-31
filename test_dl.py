#!/usr/bin/python
# -*- coding: UTF-8 -*-
# create date: 2025/2/1
# __author__: 'Alex Lu'

import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from timm.layers.classifier import ClassifierHead
import pandas as pd
from utils.datasets import *
from sklearn.model_selection import train_test_split
from utils.misc import *
from datetime import datetime
from models.titration_model import CustomModule


def get_ds(root_path):
    ds = pd.read_csv(os.path.join(root_path, 'labels.csv'), dtype=str)
    num_classes = len(ds.iloc[:, 0].unique())

    # 划分训练集和验证集
    valid_videos = ['MP_5', 'MP_17', 'VID_20250114_123113', 'VID_20250114_124017']
    test_videos = ['MM_3', 'MP_3', 'MP_19', 'VID_20250114_122658', 'VID_20250114_130314', 'video_20250114_123706']

    all_videos = ds.iloc[:, 1].unique()
    train_videos = list(set(all_videos) - set(valid_videos) - set(test_videos))

    X_train = ds[ds.iloc[:, 1].isin(train_videos)].reset_index(drop=True)
    X_valid = ds[ds.iloc[:, 1].isin(valid_videos)].reset_index(drop=True)
    X_test = ds[ds.iloc[:, 1].isin(test_videos)].reset_index(drop=True)

    print(X_train.shape, X_valid.shape, X_test.shape)

    train_ds = ImageDataset(X_train, root_path, transform)
    valid_ds = ImageDataset(X_valid, root_path, transform)
    test_ds = ImageDataset(X_test, root_path, transform)

    return num_classes, train_ds, valid_ds, test_ds


if __name__ == '__main__':
    # 设置设备（GPU或CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    # root_path = './data/MR'
    root_path = r'E:/CH/titration/out/MR'

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize(250),
        transforms.CenterCrop(224),
        # transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    num_classes, train_ds, valid_ds, test_ds = get_ds(root_path)

    # 创建数据加载器
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # 实例化模型
    model_name = 'convnext_tiny'
    model = CustomModule(model_name, num_classes=num_classes)

    kwargs = {}
    kwargs['model_name_prefix'] = f'MR_{model_name}'

    print(train_ds.__len__(), valid_ds.__len__(), test_ds.__len__())
    # 训练模型
    # best_model, results = train_predict(model, device, train_loader, valid_loader, test_loader, **kwargs)