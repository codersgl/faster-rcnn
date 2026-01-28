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
