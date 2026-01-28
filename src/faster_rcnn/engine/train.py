from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from faster_rcnn.utils.losses import rpn_loss
from faster_rcnn.utils.misc import assign_rpn_targets


def train_rpn_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device = torch.device("cpu"),
    lambda_reg: float = 1.0,
    grad_clip: float = 10.0,
) -> Dict[str, float]:
    """Train RPN for one epoch with proper batch handling.

    Args:
        model: RPN model
        dataloader: DataLoader for training data
        optimizer: Optimizer for training
        device: Device to train on
        lambda_reg: Weight for regression loss
        grad_clip: Maximum gradient norm for clipping

    Returns:
        Dictionary containing average loss metrics for the epoch
    """
    model.train()
    model.to(device)

    epoch_losses = {"total": 0.0, "cls": 0.0, "reg": 0.0}

    progress_bar = tqdm(dataloader, desc="Training RPN", leave=False)

    for batch_idx, (images, targets) in enumerate(progress_bar):
        # 1. 准备数据
        images = images.to(device)
        batch_size = images.shape[0]

        # 2. 前向传播
        cls_logits, reg_deltas, anchors = model(images)

        # 3. 准备批处理目标
        batch_gt_labels = []
        batch_reg_targets = []

        for i in range(batch_size):
            # 处理每个样本
            sample_anchors = anchors[i]  # [num_anchors, 4]
            sample_gt_boxes = targets[i]["boxes"].to(device)  # [M_i, 4]

            # 分配目标
            gt_labels, reg_targets = assign_rpn_targets(
                sample_anchors, sample_gt_boxes, device
            )

            batch_gt_labels.append(gt_labels)
            batch_reg_targets.append(reg_targets)

        # 4. 堆叠成批处理形式
        gt_labels_batch = torch.stack(
            batch_gt_labels, dim=0
        )  # [batch_size, num_anchors]
        reg_targets_batch = torch.stack(
            batch_reg_targets, dim=0
        )  # [batch_size, num_anchors, 4]

        # 5. 计算损失
        optimizer.zero_grad()
        total_loss, cls_loss, reg_loss = rpn_loss(
            cls_logits,
            reg_deltas,
            gt_labels_batch,
            reg_targets_batch,
            lambda_=lambda_reg,
        )

        # 6. 反向传播和优化
        total_loss.backward()

        # 梯度裁剪防止爆炸
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()

        # 7. 记录损失
        epoch_losses["total"] += total_loss.item()
        epoch_losses["cls"] += cls_loss.item()
        epoch_losses["reg"] += reg_loss.item()

        # 8. 更新进度条
        progress_bar.set_postfix(
            {
                "loss": f"{total_loss.item():.4f}",
                "cls": f"{cls_loss.item():.4f}",
                "reg": f"{reg_loss.item():.4f}",
            }
        )

    # 9. 计算平均损失
    num_batches = len(dataloader)
    avg_losses = {
        "total_loss": epoch_losses["total"] / num_batches,
        "cls_loss": epoch_losses["cls"] / num_batches,
        "reg_loss": epoch_losses["reg"] / num_batches,
    }

    # 10. 打印epoch总结
    print(
        f"[RPN Epoch] Total: {avg_losses['total_loss']:.4f}, "
        f"Cls: {avg_losses['cls_loss']:.4f}, Reg: {avg_losses['reg_loss']:.4f}"
    )

    return avg_losses


@torch.no_grad()
def validate_rpn(model, dataloader, device, lambda_reg=1.0):
    """Validate RPN model."""
    model.eval()
    val_losses = {"total": 0.0, "cls": 0.0, "reg": 0.0}

    for images, targets in tqdm(dataloader, desc="Validating RPN"):
        images = images.to(device)
        batch_size = images.size(0)
        cls_logits, reg_deltas, anchors = model(images)

        batch_gt_labels = []
        batch_reg_targets = []

        for i in range(batch_size):
            # 处理每个样本
            sample_anchors = anchors[i]  # [num_anchors, 4]
            sample_gt_boxes = targets[i]["boxes"].to(device)  # [M_i, 4]

            # 分配目标
            gt_labels, reg_targets = assign_rpn_targets(
                sample_anchors, sample_gt_boxes, device
            )

            batch_gt_labels.append(gt_labels)
            batch_reg_targets.append(reg_targets)

        # 4. 堆叠成批处理形式
        gt_labels_batch = torch.stack(
            batch_gt_labels, dim=0
        )  # [batch_size, num_anchors]
        reg_targets_batch = torch.stack(
            batch_reg_targets, dim=0
        )  # [batch_size, num_anchors, 4]

        total_loss, cls_loss, reg_loss = rpn_loss(
            cls_logits,
            reg_deltas,
            gt_labels_batch,
            reg_targets_batch,
            lambda_=lambda_reg,
        )
        val_losses["total"] += total_loss.item()
        val_losses["cls"] += cls_loss.item()
        val_losses["reg"] += reg_loss.item()

    num_batches = len(dataloader)
    return {k: v / num_batches for k, v in val_losses.items()}


def train_rpn_complete(
    model, train_loader, val_loader, num_epochs, device, save_dir="checkpoints"
):
    """Complete RPN training with validation and checkpointing."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_loss = float("inf")
    history = {"train": [], "val": []}

    for epoch in range(num_epochs):
        # 训练
        train_metrics = train_rpn_one_epoch(model, train_loader, optimizer, device)

        # 验证
        val_metrics = validate_rpn(model, val_loader, device)

        # 学习率调整
        scheduler.step()

        # 保存最佳模型
        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                },
                f"{save_dir}/rpn_best.pth",
            )

        # 定期保存检查点
        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                },
                f"{save_dir}/rpn_epoch_{epoch}.pth",
            )

        # 记录历史
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # 打印epoch总结
        print(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"Train Loss: {train_metrics['total_loss']:.4f}, "
            f"Val Loss: {val_metrics['total']:.4f}, "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

    return history
