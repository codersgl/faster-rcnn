from typing import Callable, Tuple

import torch
from PIL import Image
from torchvision import transforms


def get_transforms(
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Callable[[Image.Image], torch.Tensor]:
    """Get transforms for image preprocessing.

    Args:
        mean (Tuple[float, float, float]): Mean values for normalization.
        std (Tuple[float, float, float]): Standard deviation values for normalization.

    Returns:
        Callable[[Image.Image], torch.Tensor]: A callable that applies the transforms to an image.
    """

    def resize_shorter_side(image: Image.Image) -> Image.Image:
        """Resize image so that the shorter side is 600 pixels long."""
        width, height = image.size

        if width < height:
            new_width = 600
            new_height = int(height * 600 / width)
        else:
            new_height = 600
            new_width = int(width * 600 / height)

        new_width = int(new_width)
        new_height = int(new_height)

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return transforms.Compose(
        [
            resize_shorter_side,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
