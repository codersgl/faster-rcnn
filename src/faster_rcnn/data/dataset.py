import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from faster_rcnn.utils.data import get_transform


class VocDataset(Dataset):
    CLASS_TO_IDX = {
        "dog": 1,
        "sheep": 2,
        "horse": 3,
        "tvmonitor": 4,
        "sofa": 5,
        "bird": 6,
        "cow": 7,
        "bottle": 8,
        "motorbike": 9,
        "cat": 10,
        "bus": 11,
        "pottedplant": 12,
        "bicycle": 13,
        "car": 14,
        "aeroplane": 15,
        "chair": 16,
        "train": 17,
        "diningtable": 18,
        "boat": 19,
        "person": 20,
    }

    def __init__(
        self,
        root_dir: Path,
        is_train: bool = True,
        transform=None,
        keep_difficult: bool = False,
    ) -> None:
        """
        VOC Dataset 初始化
        Args:
            root_dir (Path): 数据集根目录
            is_train (bool): 是否为训练集
            transform (callable, optional): 图片转换函数
            keep_difficult (bool): 是否保留标记为 difficult 的目标 (默认 False)
        """
        super().__init__()
        self.root_dir = root_dir
        self.is_train = is_train
        self.transform = transform
        self.keep_difficult = keep_difficult

        self.image_ids = self._load_image_ids()

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Tensor]]:
        image_id = self.image_ids[idx]

        image_path = self.root_dir / "JPEGImages" / f"{image_id}.jpg"
        xml_path = self.root_dir / "Annotations" / f"{image_id}.xml"

        image = Image.open(image_path).convert("RGB")
        target = self._parse_xml(xml_path)

        target["image_id"] = torch.tensor([idx])

        if self.transform:
            original_size = image.size
            image = self.transform(image)
            target = self._resize_boxes(target, original_size, image.shape[-2:])

        return image, target

    def _load_image_ids(self) -> List[str]:
        subset = "train" if self.is_train else "val"
        image_set_file = self.root_dir / "ImageSets" / "Main" / f"{subset}.txt"

        if not image_set_file.exists():
            raise FileNotFoundError(f"Image set file not found: {image_set_file}")

        with open(image_set_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f]

    def _parse_xml(self, xml_path: Path) -> Dict[str, Tensor]:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for obj in root.iter("object"):
            difficult = obj.find("difficult")
            is_difficult = int(difficult.text) if difficult is not None else 0  # type: ignore

            if is_difficult and not self.keep_difficult:
                continue

            name = obj.find("name").text  # type: ignore
            if name not in self.CLASS_TO_IDX:
                continue

            labels.append(self.CLASS_TO_IDX[name])
            iscrowd.append(is_difficult)

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)  # type: ignore
            ymin = float(bndbox.find("ymin").text)  # type: ignore
            xmax = float(bndbox.find("xmax").text)  # type: ignore
            ymax = float(bndbox.find("ymax").text)  # type: ignore

            # 存储 box [xmin, ymin, xmax, ymax]
            boxes.append([xmin, ymin, xmax, ymax])
            areas.append((xmax - xmin) * (ymax - ymin))

        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        return {
            "boxes": boxes,
            "labels": labels,
            "area": areas,
            "iscrowd": iscrowd,
        }

    def _resize_boxes(
        self,
        target: Dict[str, Tensor],
        original_size: Tuple[int, int],
        new_size: Tuple[int, int],
    ) -> Dict[str, Tensor]:
        if len(target["boxes"] == 0):
            return target

        original_width, original_height = original_size
        new_width, new_height = new_size

        width_scale = new_width / original_width
        height_scale = new_height / original_height

        boxes = target["boxes"].clone()

        boxes[:, 0] = boxes[:, 0] * width_scale
        boxes[:, 1] = boxes[:, 1] * height_scale
        boxes[:, 2] = boxes[:, 2] * width_scale
        boxes[:, 3] = boxes[:, 3] * height_scale

        target["boxes"] = boxes
        target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        return target


if __name__ == "__main__":
    try:
        train_dataset = VocDataset(
            Path("data/raw/VOCdevkit2007/VOC2007"),
            is_train=True,
            transform=get_transform(is_train=True),
        )
        print(f"Dataset length: {len(train_dataset)}")

        image, target = next(iter(train_dataset))
        print(f"Image tensor shape: {image.shape}")
        print("Target keys:", target.keys())
        print("Boxes sample:", target["boxes"])
        print("Labels sample:", target["labels"])
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
