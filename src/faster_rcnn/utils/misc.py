from pathlib import Path
from typing import Dict

import json


def get_class_map(path: Path) -> Dict:
    class_map_idx = {}
    id = 1

    for file in path.iterdir():
        if file.is_dir():
            continue
        if "_train" in file.name:
            class_name, _ = file.name.split("_")

            if class_name not in class_map_idx:
                class_map_idx[class_name] = id
                id += 1

    with open(path.parent.parent / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_map_idx, f)

    return class_map_idx


def train_one_epoch(model, train_dataloader, loss_fn, device="cpu"):
    model.train()
    model.to(device)

    for image, target in train_dataloader:
        image = image.to(device)
        labels = target["labels"].to(device)
        boxes = target["boxes"].to(device)
        output = model(image)
        loss = loss_fn(output, [labels, boxes])


if __name__ == "__main__":
    path = Path("data/raw/VOCdevkit2007/VOC2007/ImageSets/Main")
    get_class_map(path)
