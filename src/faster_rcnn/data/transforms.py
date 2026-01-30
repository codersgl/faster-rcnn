from PIL import Image
from torchvision import transforms


def resize_image(image: Image.Image) -> Image.Image:
    """Resize image according to Faster R-CNN paper: shorter side=600, longer side≤1000"""
    width, height = image.size

    # 计算缩放比例
    scale = 600.0 / min(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 确保最长边不超过1000
    if max(new_width, new_height) > 1000:
        scale = 1000.0 / max(new_width, new_height)
        new_width = int(new_width * scale)
        new_height = int(new_height * scale)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def get_transforms(
    train: bool = True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
):
    transforms_list = [resize_image]

    if train:
        transforms_list.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

    transforms_list.extend(
        [
            transforms.ToTensor(),  # type: ignore
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return transforms.Compose(transforms_list)
