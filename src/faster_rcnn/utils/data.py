from torchvision import transforms


def get_transform(is_train: bool = True):
    transforms_list = []
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    )
    return transforms.Compose(transforms_list)
