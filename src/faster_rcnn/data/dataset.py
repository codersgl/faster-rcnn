import json
from pathlib import Path
from typing import Any, Dict, Tuple
from xml.etree import ElementTree as ET

import torch
from PIL import Image
from torch.utils.data import Dataset

from faster_rcnn.data.transforms import get_transforms


class PascalVOC(Dataset):
    def __init__(self, root_dir: str | Path, train: bool, transform=None):
        """
        Initialize the PascalVOC dataset.

        Args:
            root_dir (Path): Root directory of the dataset.

            train (bool): Whether to load the training or validation set.

            transform (callable, optional): Optional transform to be applied on the image.
        """

        self.root_dir = Path(root_dir) if isinstance(root_dir, str) else root_dir
        self.transform = transform

        # Load class names from JSON file
        with open(self.root_dir / "class_names.json", "r", encoding="utf-8") as f:
            self.class_names = json.load(f)

        # Load image paths from text file
        text_file = (
            self.root_dir / "ImageSets/Main/train.txt"
            if train
            else self.root_dir / "ImageSets/Main/val.txt"
        )
        with open(text_file, "r", encoding="utf-8") as f:
            self.image_paths = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get item from dataset.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            tuple: Tuple containing the image and target dictionary.
        """
        image_path: Path = (
            self.root_dir / "JPEGImages" / (self.image_paths[idx] + ".jpg")
        )
        image = Image.open(image_path).convert("RGB")

        original_width, original_height = image.size

        # Initialize target dictionary
        target = {}
        boxes, labels = self._parse_xml(
            self.root_dir / "Annotations" / (self.image_paths[idx] + ".xml")
        )
        target["boxes"] = torch.tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        target["image_id"] = torch.tensor([idx], dtype=torch.int64)

        if self.transform:
            image = self.transform(image)
            width, height = image.size
            target["boxes"] = target["boxes"] * torch.tensor(
                [width, height, width, height], dtype=torch.float32
            )
        else:
            image = torch.tensor(image, dtype=torch.float32)

        return image, target

    def _parse_xml(self, xml_path: Path):
        """
        Parse XML file and extract bounding boxes and labels.

        Args:
            xml_path (Path): Path to the XML file.

        Returns:
            tuple: Tuple containing bounding boxes and labels.
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []
        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)  # type: ignore
            ymin = int(bbox.find("ymin").text)  # type: ignore
            xmax = int(bbox.find("xmax").text)  # type: ignore
            ymax = int(bbox.find("ymax").text)  # type: ignore
            boxes.append([xmin, ymin, xmax, ymax])

            label = obj.find("name").text  # type: ignore
            labels.append(self.class_names[label])

        return boxes, labels


if __name__ == "__main__":
    # test
    data_path = Path("data/raw/VOCdevkit2007/VOC2007")
    dataset = PascalVOC(
        root_dir=data_path,
        train=True,
        transform=get_transforms(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    )

    sample = dataset[0]
    print(sample)
