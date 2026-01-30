import os
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from faster_rcnn.data.collate import collate_fn
from faster_rcnn.data.dataset import PascalVOC
from faster_rcnn.data.transforms import get_transforms
from faster_rcnn.models.faster_rcnn import FasterRCNN
from faster_rcnn.utils.metrics import calculate_map


@torch.no_grad()
def validate(
    model: FasterRCNN,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    iou_threshold: float = 0.5,
):
    model.eval()
    logger.info("Starting validation...")

    det_boxes = []
    det_labels = []
    det_scores = []
    gt_boxes = []
    gt_labels = []

    for images, targets in tqdm(dataloader, desc="Validation"):
        images = images.to(device)

        # Predict
        predictions = model.predict(images)

        # Collect results
        for i, pred in enumerate(predictions):
            det_boxes.append(pred["boxes"].cpu())
            det_labels.append(pred["labels"].cpu())
            det_scores.append(pred["scores"].cpu())

            gt_boxes.append(targets[i]["boxes"])
            gt_labels.append(targets[i]["labels"])

    # Calculate mAP
    metrics = calculate_map(
        det_boxes=det_boxes,
        det_labels=det_labels,
        det_scores=det_scores,
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
        iou_threshold=iou_threshold,
        num_classes=num_classes,
        background_class_id=0,
    )

    return metrics


@hydra.main(
    version_base=None, config_path="../src/faster_rcnn/configs", config_name="config"
)
def main(cfg: DictConfig):
    # Setup paths
    root_dir = Path(hydra.utils.get_original_cwd())
    data_path = root_dir / cfg.data.root_dir

    # Output directory (managed by Hydra)
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    logger.add(output_dir / "val.log", rotation="10 MB")
    logger.info(f"Output directory: {output_dir}")

    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    if "checkpoint" not in cfg:
        logger.warning(
            "No checkpoint provided! Use +checkpoint=/path/to/model.pth to evaluate a trained model."
        )

    # Setup Device
    device = torch.device(
        cfg.environment.device if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Create DataLoaders
    val_dataset = PascalVOC(
        root_dir=data_path,
        train=False,
        transform=get_transforms(train=False, mean=cfg.data.mean, std=cfg.data.std),
        class_names=cfg.data.class_names,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.environment.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    num_classes = len(val_dataset.class_names)
    logger.info(f"Number of classes (including background): {num_classes}")

    # Create Model
    model = FasterRCNN(
        num_classes=num_classes,
        backbone_channels=512,  # VGG16
        rpn_mid_channels=512,
        num_anchors=len(cfg.model.rpn.anchor_sizes) * len(cfg.model.rpn.aspect_ratios),
        im_size=(1000, 600),
        device=device,
    )
    model.to(device)

    # Load Checkpoint
    if "checkpoint" in cfg and os.path.exists(cfg.checkpoint):
        logger.info(f"Loading checkpoint from {cfg.checkpoint}")
        checkpoint = torch.load(cfg.checkpoint, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        logger.info("Evaluating with random weights (untrained model).")

    # Run Validation
    metrics = validate(
        model=model,
        dataloader=val_loader,
        device=device,
        num_classes=num_classes,
        iou_threshold=0.5,
    )

    logger.info("Validation Results:")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
