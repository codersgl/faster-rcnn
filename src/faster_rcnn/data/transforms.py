from typing import Callable, Tuple

from PIL import Image
from torchvision import transforms


def get_transforms(
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Callable[[Image.Image], Image.Image]:
    """Get transforms for image preprocessing.

    Args:
        mean (Tuple[float, float, float]): Mean values for normalization.
        std (Tuple[float, float, float]): Standard deviation values for normalization.

    Returns:
        Callable[[Image.Image], Image.Image]: A callable that applies the transforms to an image.
    """

    # Image are resized so that the shorter side is 600 pixels long
    def resize_shorter_side(image: Image.Image) -> Image.Image:
        if image.width > image.height:
            return image.resize((int(image.width * 600 / image.height), 600))
        else:
            return image.resize((600, int(image.height * 600 / image.width)))

    return transforms.Compose([
        resize_shorter_side,
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
