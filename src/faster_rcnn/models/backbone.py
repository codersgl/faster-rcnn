"""Shared Convolutional Backbone: VGG-16: 13 convolutional layers (more commonly used)"""

import torch.nn as nn
from torchvision import models


def vgg16_backbone():
    """VGG-16 backbone"""
    backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features

    # Optimize memory: set ReLU to inplace=True
    for layer in backbone:
        if isinstance(layer, nn.ReLU):
            layer.inplace = True

    shared_backbone = nn.Sequential(*list(backbone.children())[:-1])

    # Freeze first 4 conv layers (conv1_1 through conv2_2) to match original paper
    # These are the convolutional layers at indices: 0, 2, 5, 7
    conv_indices_to_freeze = [0, 2, 5, 7]

    for idx in conv_indices_to_freeze:
        layer = shared_backbone[idx]
        if isinstance(layer, nn.Conv2d):
            for param in layer.parameters():
                param.requires_grad = False

    return shared_backbone
