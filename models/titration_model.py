#!/usr/bin/python
# -*- coding: UTF-8 -*-
# create date: 2025/1/13
# __author__: 'Alex Lu'
import timm
import torch.nn as nn


class CustomModule(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=False):
        super(CustomModule, self).__init__()

        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        return self.model(x)