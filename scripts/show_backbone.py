#!.venv/bin/python3
from typing import Sequence

import torch
from torchinfo import summary

from faster_rcnn.models.backbone import vgg16_backbone


def show_backbone(
    model: torch.nn.Module, size: Sequence[int], device: torch.device
) -> None:
    x = torch.randn(size, device=device)
    summary(model, input_data=x)


if __name__ == "__main__":
    backbone = vgg16_backbone()
    show_backbone(backbone, (3, 224, 224), device=torch.device("cpu"))
