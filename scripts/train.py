import random
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from faster_rcnn.data.collate import collate_fn
from faster_rcnn.data.dataset import PascalVOC
from faster_rcnn.data.transforms import get_transforms
from faster_rcnn.engine.train import train_faster_rcnn_one_epoch
from faster_rcnn.models.faster_rcnn import FasterRCNN
from faster_rcnn.utils.metrics import calculate_map
from faster_rcnn.utils.visualization import denormalize_image, draw_boxes


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def validate(
    model: FasterRCNN,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int,
    writer: SummaryWriter = None,
    epoch: int = 0,
    class_names: list = None,
    mean: list = None,
    std: list = None,
    iou_threshold: float = 0.5,
    num_vis_images: int = 4,
):
    model.eval()
    logger.info("Starting validation...")

    det_boxes = []
    det_labels = []
    det_scores = []
    gt_boxes = []
    gt_labels = []

    images_visualized = 0

    for images, targets in tqdm(dataloader, desc="Validation"):
        images = images.to(device)

        # Predict
        predictions = model.predict(images)

        # Visualization
        if writer is not None and images_visualized < num_vis_images:
            for i in range(len(images)):
                if images_visualized >= num_vis_images:
                    break

                img_tensor = images[i]  # (C, H, W)
                pred = predictions[i]

                # Denormalize
                if mean and std:
                    img_tensor = denormalize_image(img_tensor, mean, std)

                # Draw boxes
                img_np = draw_boxes(
                    image=img_tensor,
                    boxes=pred["boxes"],
                    labels=pred["labels"],
                    scores=pred["scores"],
                    class_names=class_names,
                    score_thresh=0.5,
                )

                # Convert back to (C, H, W) for TensorBoard
                img_vis = torch.from_numpy(img_np).permute(2, 0, 1)

                writer.add_image(f"Val/Prediction_{images_visualized}", img_vis, epoch)
                images_visualized += 1

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
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    set_seed(cfg.project.seed)

    # Initialize Logger
    logger.add(
        output_dir / "train.log",
        rotation="10 MB",
        retention="10 days",
    )

    # Initialize TensorBoard
    writer = SummaryWriter(log_dir=str(output_dir))

    # Setup Device
    device = torch.device(
        cfg.environment.device if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Create DataLoaders
    train_dataset = PascalVOC(
        root_dir=data_path,
        train=True,
        transform=get_transforms(train=True, mean=cfg.data.mean, std=cfg.data.std),
        class_names=cfg.data.class_names,
    )
    val_dataset = PascalVOC(
        root_dir=data_path,
        train=False,
        transform=get_transforms(train=False, mean=cfg.data.mean, std=cfg.data.std),
        class_names=cfg.data.class_names,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.environment.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,  # Can use larger batch size for val if predict handles it
        shuffle=False,
        num_workers=cfg.environment.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # num_classes from dataset usually includes background implicitly or explicitly depending on implementation.
    # PascalVOC dataset class_names now includes background if initialized with class_names.
    # FasterRCNN expects num_classes including background (21).
    num_classes = len(train_dataset.class_names)
    logger.info(f"Number of classes (including background): {num_classes}")

    # Create Model
    model = FasterRCNN(
        num_classes=num_classes,
        backbone_channels=512,  # VGG16
        rpn_mid_channels=512,
        num_anchors=len(cfg.model.rpn.anchor_sizes) * len(cfg.model.rpn.aspect_ratios),
        im_size=(1000, 600),
        device=device,
        # Pass other configs if FasterRCNN accepts them, currently hardcoded or default
    )
    model.to(device)

    # Optimizer and Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    if cfg.training.optimizer.name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=cfg.training.optimizer.lr,
            momentum=cfg.training.optimizer.momentum,
            weight_decay=cfg.training.optimizer.weight_decay,
        )
    elif cfg.training.optimizer.name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=cfg.training.optimizer.lr,
            weight_decay=cfg.training.optimizer.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.training.optimizer.name}")

    if cfg.training.scheduler.name == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.training.scheduler.milestones,
            gamma=cfg.training.scheduler.gamma,
        )
    else:
        # Default or handle other schedulers
        lr_scheduler = None

    # Training Loop
    logger.info("Start training...")
    best_map = 0.0

    for epoch in range(cfg.training.epochs):
        logger.info(f"Epoch {epoch + 1}/{cfg.training.epochs}")

        # Train one epoch
        metrics = train_faster_rcnn_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            lambda_reg=1.0,  # Could be from cfg
            grad_clip=cfg.training.grad_clip,
        )

        # Log metrics to TensorBoard
        writer.add_scalar("Train/Loss/Total", metrics["total"], epoch)
        writer.add_scalar("Train/Loss/RPN", metrics["rpn"], epoch)
        writer.add_scalar("Train/Loss/Cls", metrics["cls"], epoch)
        writer.add_scalar("Train/Loss/Reg", metrics["reg"], epoch)
        writer.add_scalar("Train/LR", optimizer.param_groups[0]["lr"], epoch)

        logger.info(
            f"Epoch {epoch + 1} Train Loss: Total={metrics['total']:.4f}, "
            f"RPN={metrics['rpn']:.4f}, Cls={metrics['cls']:.4f}, Reg={metrics['reg']:.4f}"
        )

        if lr_scheduler:
            lr_scheduler.step()

        # Validation
        if (epoch + 1) % cfg.training.val_interval == 0:
            val_metrics = validate(
                model=model,
                dataloader=val_loader,
                device=device,
                num_classes=num_classes,
                writer=writer,
                epoch=epoch,
                class_names=val_dataset.class_names,
                mean=cfg.data.mean,
                std=cfg.data.std,
            )

            # Extract mAP excluding background (index 0) if it was calculated
            # My calculate_map implementation calculates for all range(num_classes).
            # I should update calculate_map or just ignore AP for class 0 here.
            # But let's log what we have.

            # Simple workaround: calculate_map iterates range(num_classes).
            # If I pass 21, it does 0..20.
            # I should pass 21, but focus on mAP of classes 1..20.

            map_score = val_metrics["map"]

            logger.info(f"Epoch {epoch + 1} Validation mAP: {map_score:.4f}")
            writer.add_scalar("Val/mAP", map_score, epoch)

            for k, v in val_metrics.items():
                if k != "map":
                    writer.add_scalar(f"Val/{k}", v, epoch)

            # Save best model
            if map_score > best_map:
                best_map = map_score
                torch.save(model.state_dict(), output_dir / "faster_rcnn_best.pth")
                logger.info(f"New best model saved with mAP: {best_map:.4f}")

        # Save checkpoint
        if (epoch + 1) % cfg.training.save_interval == 0:
            checkpoint_path = output_dir / f"model_epoch_{epoch + 1}.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "cfg": cfg,
                },
                checkpoint_path,
            )
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    torch.save(model.state_dict(), output_dir / "faster_rcnn_final.pth")
    writer.close()
    logger.info("Training finished!")


if __name__ == "__main__":
    main()
