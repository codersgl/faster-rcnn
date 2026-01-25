"""Shared Convolutional Backbone: VGG-16: 13 convolutional layers (more commonly used)"""

import torch.nn as nn
from torchvision import models


def vgg16_backbone():
    """VGG-16 backbone"""
    backbone = models.vgg16(pretrained=True).features
    shared_backbone = nn.Sequential(*list(backbone.children())[:-1])

    # Freeze first 10 conv layers (conv1_1 through conv4_3)
    # These are the convolutional layers at indices: 0, 2, 5, 7, 10, 12, 14, 17, 19, 21
    conv_indices_to_freeze = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21]

    for idx in conv_indices_to_freeze:
        layer = shared_backbone[idx]
        if isinstance(layer, nn.Conv2d):
            for param in layer.parameters():
                param.requires_grad = False

    return shared_backbone
