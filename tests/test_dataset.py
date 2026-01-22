import pytest

from faster_rcnn.data.dataset import PascalVOC
from faster_rcnn.data.transforms import get_transforms


# Test src/faster_rcnn/data/dataset.py
@pytest.mark.parametrize("root_dir", ["data/raw/VOCdevkit2007/VOC2007"])
@pytest.mark.parametrize("train", [True, False])
@pytest.mark.parametrize("transform", [get_transforms()])
def test_pascal_voc_dataset(root_dir, train, transform):
    dataset = PascalVOC(
        root_dir=root_dir,
        train=train,
        transform=transform,
    )

    image, target = dataset[0]

    width, height = image.size(1), image.size(2)
    short_size = min(width, height)

    # Images are resized so that their shorter side is s = 600 pixels
    assert short_size == 600

    assert target["boxes"].size(1) == 4

    assert target["labels"].size(0) == target["boxes"].size(0)
