import json
from pathlib import Path
from typing import Dict

from loguru import logger

# config log
logger.add("logs/app.log", rotation="10 MB", retention="10 days")


def get_categories_save_to_json_file(dataset_path: Path) -> Dict:
    """Get class names from a dataset and save them to a JSON file.

    Args:
        dataset_path (Path): Path to the dataset directory.

    Returns:
        Dict: A dictionary mapping class names to their corresponding indices.
    example:
        dataset_path = Path("path/to/dataset")
        get_categories_save_to_json_file(dataset_path)
    """

    save_json_path: Path = dataset_path / "categories.json"
    categories: dict = {}

    # Get class names from dataset/ImageSets/Main/*_train.txt, that * is one class name.
    categories_dir = dataset_path / "ImageSets" / "Main"
    for file in categories_dir.glob("*_train.txt"):
        class_name = file.stem.split("_")[0]
        if class_name not in categories:
            categories[class_name] = len(categories)

    # Sort class names by index
    categories = dict(sorted(categories.items(), key=lambda item: item[1]))

    logger.info(f"Class names: {categories}")

    # Save class names to JSON file
    with open(save_json_path, "w", encoding="utf-8") as f:
        json.dump(categories, f)

    return categories


if __name__ == "__main__":
    # test get_categories_save_to_json_file
    # dataset_path: data/raw/VOCdevkit2007/VOC2007
    dataset_path = Path("data/raw/VOCdevkit2007/VOC2007")
    get_categories_save_to_json_file(dataset_path)
