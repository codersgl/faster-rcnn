from torchvision import transforms

from typing import Any
from PIL import Image


class FixedShortSideResize:
    def __init__(self, short_side=600) -> None:
        self.short_side = short_side

    def __call__(self, image) -> Any:
        width, height = image.size

        if width < height:
            new_width = self.short_side
            new_hight = int(height * (self.short_side / width))
        else:
            new_hight = self.short_side
            new_width = int(width * (self.short_side / height))

        return image.resize((new_width, new_hight), Image.Resampling.BILINEAR)


def get_transform(is_train: bool = True, short_side: int = 600):
    transforms_list = []

    transforms_list.append(FixedShortSideResize(short_side=short_side))
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    return transforms.Compose(transforms_list)
