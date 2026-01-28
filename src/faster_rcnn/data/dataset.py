import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from xml.etree import ElementTree as ET

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from faster_rcnn.data.transforms import get_transforms


class PascalVOC(Dataset):
    def __init__(
        self,
        root_dir: Union[str, Path],
        train: bool,
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the PascalVOC dataset.

        Args:
            root_dir: Root directory of the dataset.
            train: Whether to load the training or validation set.
            transform: Optional transform to be applied on the image.
        """
        self.root_dir = Path(root_dir) if isinstance(root_dir, str) else root_dir
        self.transform = transform
        self.train = train

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")

        # Load class names from JSON file
        class_names_path = self.root_dir / "class_names.json"
        if not class_names_path.exists():
            raise FileNotFoundError(f"Class names file not found: {class_names_path}")

        with open(class_names_path, "r", encoding="utf-8") as f:
            self.class_names = json.load(f)

        # 创建从类别名到索引的映射
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Load image paths from text file
        text_file = (
            self.root_dir / "ImageSets/Main/train.txt"
            if train
            else self.root_dir / "ImageSets/Main/val.txt"
        )

        if not text_file.exists():
            raise FileNotFoundError(f"Image list file not found: {text_file}")

        with open(text_file, "r", encoding="utf-8") as f:
            self.image_paths = [line.strip() for line in f.readlines() if line.strip()]

        self._validate_files()

    def _validate_files(self):
        """验证所有图像和标注文件是否存在"""
        missing_files = []

        for img_name in self.image_paths:
            # 检查图像文件
            img_path = self.root_dir / "JPEGImages" / (img_name + ".jpg")
            if not img_path.exists():
                missing_files.append(str(img_path))

            # 检查标注文件
            xml_path = self.root_dir / "Annotations" / (img_name + ".xml")
            if not xml_path.exists():
                missing_files.append(str(xml_path))

        if missing_files:
            print(f"警告: 找到 {len(missing_files)} 个缺失文件")
            if len(missing_files) <= 10:
                for f in missing_files[:10]:
                    print(f"  - {f}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Get item from dataset.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            Tuple containing the image tensor and target dictionary.
        """
        img_name = self.image_paths[idx]

        image_path = self.root_dir / "JPEGImages" / (img_name + ".jpg")
        image = Image.open(image_path).convert("RGB")

        original_width, original_height = image.size

        xml_path = self.root_dir / "Annotations" / (img_name + ".xml")
        boxes, labels = self._parse_xml(xml_path)

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)

        areas = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (
            boxes_tensor[:, 3] - boxes_tensor[:, 1]
        )
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": areas,
            "iscrowd": iscrowd,
            "orig_size": torch.tensor(
                [original_height, original_width], dtype=torch.int64
            ),
        }

        if self.transform:
            image_trans, target_trans = self._apply_transform_with_boxes(image, target)
            return image_trans, target_trans
        else:
            image_tensor = (
                torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            )
            return image_tensor, target

    def _apply_transform_with_boxes(
        self, image: Image.Image, target: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        应用transform并相应地调整边界框。
        """
        orig_width, orig_height = image.size

        # transform is not None here because it is checked in __getitem__
        # but check again for safety or just use it.
        if self.transform:
            image_transformed = self.transform(image)
        else:
            image_transformed = image

        if isinstance(image_transformed, torch.Tensor):
            new_height, new_width = (
                image_transformed.shape[1],
                image_transformed.shape[2],
            )
            image_tensor = image_transformed
        else:
            # If transform didn't return a tensor (e.g. only resize), convert it here.
            new_width, new_height = image_transformed.size
            image_tensor = (
                torch.from_numpy(np.array(image_transformed)).permute(2, 0, 1).float()
                / 255.0
            )

        scale_x = new_width / orig_width
        scale_y = new_height / orig_height

        boxes = target["boxes"].clone()
        boxes[:, 0] *= scale_x  # xmin
        boxes[:, 1] *= scale_y  # ymin
        boxes[:, 2] *= scale_x  # xmax
        boxes[:, 3] *= scale_y  # ymax

        target["boxes"] = boxes
        target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        target["size"] = torch.tensor([new_height, new_width], dtype=torch.int64)

        return image_tensor, target

    def _parse_xml(self, xml_path: Path) -> Tuple[List[List[float]], List[int]]:
        """
        Parse XML file and extract bounding boxes and labels.

        Args:
            xml_path: Path to the XML file.

        Returns:
            Tuple containing bounding boxes and labels.
        """
        if not xml_path.exists():
            raise FileNotFoundError(f"XML file not found: {xml_path}")

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            boxes = []
            labels = []

            for obj in root.findall("object"):
                difficult = obj.find("difficult")
                if difficult is not None and difficult.text == "1":
                    continue

                bbox = obj.find("bndbox")
                if bbox is None:
                    continue

                try:
                    xmin = float(bbox.find("xmin").text)  # type: ignore
                    ymin = float(bbox.find("ymin").text)  # type: ignore
                    xmax = float(bbox.find("xmax").text)  # type: ignore
                    ymax = float(bbox.find("ymax").text)  # type: ignore

                    if xmin >= xmax or ymin >= ymax:
                        print(
                            f"警告: 无效边界框 {xmin},{ymin},{xmax},{ymax} 在 {xml_path}"
                        )
                        continue

                    boxes.append([xmin, ymin, xmax, ymax])

                    name_elem = obj.find("name")
                    if name_elem is None:
                        print(f"警告: 对象没有名称在 {xml_path}")
                        continue

                    label_name = name_elem.text
                    if label_name not in self.class_to_idx:
                        print(f"警告: 未知类别 '{label_name}' 在 {xml_path}")
                        continue

                    labels.append(self.class_to_idx[label_name])

                except (AttributeError, ValueError) as e:
                    print(f"警告: 解析边界框时出错在 {xml_path}: {e}")
                    continue

            return boxes, labels

        except ET.ParseError as e:
            raise RuntimeError(f"解析XML文件失败 {xml_path}: {e}")

    def get_class_names(self) -> List[str]:
        """获取类别名称列表"""
        return self.class_names

    def get_class_to_idx(self) -> Dict[str, int]:
        """获取类别到索引的映射"""
        return self.class_to_idx.copy()


if __name__ == "__main__":
    # test
    data_path = Path("data/raw/VOCdevkit2007/VOC2007")
    dataset = PascalVOC(
        root_dir=data_path,
        train=True,
        transform=get_transforms(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    )

    # Note: Requires dataset to exist to run
    # sample = dataset[0]
    # print(sample)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    print(type(dataloader))
