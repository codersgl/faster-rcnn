from typing import Tuple

import torch
import torch.nn.functional as F


def rpn_loss(cls_logits, reg_deltas, gt_labels, gt_reg_targets, lambda_=1.0):
    """
    Calculate RPN loss (vectorized version)

    Args:
        cls_logits: [batch_size, num_anchors, 2]
        reg_deltas: [batch_size, num_anchors, 4]
        gt_labels: [batch_size, num_anchors] (1: foreground, 0: background, -1: ignore)
        gt_reg_targets: [batch_size, num_anchors, 4]
        lambda_: weight for regression loss

    Returns:
        total_loss: scalar tensor
        cls_loss: scalar tensor
        reg_loss: scalar tensor
    """
    # Flatten all tensors
    # [batch_size, num_anchors, ...] -> [batch_size * num_anchors, ...]
    cls_logits = cls_logits.flatten(0, 1)  # [B*N, 2]
    reg_deltas = reg_deltas.flatten(0, 1)  # [B*N, 4]
    gt_labels = gt_labels.flatten()  # [B*N]
    gt_reg_targets = gt_reg_targets.flatten(0, 1)  # [B*N, 4]

    # 1. Classification Loss
    # Only compute loss for valid samples (label >= 0)
    valid_mask = gt_labels >= 0
    if valid_mask.sum() > 0:
        # F.cross_entropy expects class indices as targets (LongTensor)
        cls_loss = F.cross_entropy(
            cls_logits[valid_mask], gt_labels[valid_mask], reduction="mean"
        )
    else:
        cls_loss = torch.tensor(0.0, device=cls_logits.device)

    # 2. Regression Loss
    # Only compute loss for positive samples (label == 1)
    pos_mask = gt_labels == 1
    if pos_mask.sum() > 0:
        reg_loss = F.smooth_l1_loss(
            reg_deltas[pos_mask], gt_reg_targets[pos_mask], reduction="mean"
        )
    else:
        reg_loss = torch.tensor(0.0, device=reg_deltas.device)

    # Total loss
    total_loss = cls_loss + lambda_ * reg_loss

    return total_loss, cls_loss, reg_loss


def faster_rcnn_loss(
    cls_logits: torch.Tensor,
    bbox_preds: torch.Tensor,
    bbox_targets: torch.Tensor,
    gt_labels: torch.Tensor,
    num_classes: int,
    lambda_: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Args:
        cls_logits: (batch_size, num_rois, num_classes)
        bbox_preds: (batch_size, num_rois, num_classes * 4)
        bbox_targets: (batch_size, num_rois, 4)
        gt_labels: (batch_size, num_rois)  # 0表示背景，>0表示类别索引
        num_classes: 包括背景的总类别数
    """
    # 1. 分类损失
    if cls_logits.dim() > 2:
        cls_logits = cls_logits.flatten(0, 1)
        gt_labels = gt_labels.flatten()
        bbox_preds = bbox_preds.flatten(0, 1)
        bbox_targets = bbox_targets.flatten(0, 1)

    cls_loss = F.cross_entropy(cls_logits, gt_labels)

    # 2. 回归损失（仅对正样本计算）
    pos_mask = gt_labels > 0  # 正样本掩码

    if pos_mask.sum() > 0:
        # 选择对应类别的回归预测
        bbox_preds_pos = bbox_preds[pos_mask]  # [num_pos, num_classes*4]
        bbox_targets_pos = bbox_targets[pos_mask]  # [num_pos, 4]
        labels_pos = gt_labels[pos_mask]  # [num_pos]

        # 重塑为 [num_pos, num_classes, 4]
        bbox_preds_pos = bbox_preds_pos.view(-1, num_classes, 4)

        # 选择对应类别的预测
        batch_indices = torch.arange(len(labels_pos), device=bbox_preds.device)
        bbox_preds_selected = bbox_preds_pos[batch_indices, labels_pos]

        reg_loss = F.smooth_l1_loss(
            bbox_preds_selected, bbox_targets_pos, reduction="mean"
        )
    else:
        reg_loss = torch.tensor(0.0, device=cls_logits.device)

    # 3. 总损失
    total_loss = cls_loss + lambda_ * reg_loss

    return total_loss, cls_loss, reg_loss
