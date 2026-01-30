import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from loguru import logger

from faster_rcnn.utils.boxes import box_iou, encode_boxes

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
    categories: dict = {"__background__": 0}

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


def assign_rpn_targets(
    anchors: torch.Tensor,
    gt_boxes: torch.Tensor,
    device: torch.types.Device,
    pos_iou_thresh: float = 0.7,
    neg_iou_thresh: float = 0.3,
    min_pos_iou: float = 0.3,
    total_sample_size: int = 256,
    pos_fraction: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """为RPN分配标签和回归目标，包含正负样本采样

    Args:
        anchors: 锚框 [N, 4]
        gt_boxes: 真实框 [M, 4]
        device: 设备
        pos_iou_thresh: 正样本IoU阈值
        neg_iou_thresh: 负样本IoU阈值
        min_pos_iou: 确保每个GT至少有一个正样本的阈值
        total_sample_size: 每张图像采样的总样本数 (默认256)
        pos_fraction: 正样本占总样本的最大比例 (默认0.5)

    Returns:
        labels: 标签 [N] (1:前景, 0:背景, -1:忽略)
        reg_targets: 回归目标 [N, 4] (只有正样本有有效值)
    """
    # 1. 计算IoU矩阵
    iou_matrix = box_iou(anchors, gt_boxes)  # [N, M]

    # 2. 每个锚框的最大IoU和对应的gt索引
    max_iou_per_anchor, gt_idx_per_anchor = iou_matrix.max(dim=1)  # [N, ]

    # 3. 每个真实框的最大IoU（用于确保每个gt至少有一个正样本）
    max_iou_per_gt, _ = iou_matrix.max(dim=0)  # [M, ]

    # 4. 初始化标签和回归目标
    labels = torch.full((anchors.shape[0],), -1, dtype=torch.long, device=device)
    reg_targets = torch.zeros((anchors.shape[0], 4), device=device)

    # 5. 正样本：IoU >= pos_iou_thresh
    pos_mask = max_iou_per_anchor >= pos_iou_thresh
    labels[pos_mask] = 1

    # 6. 确保每个真实框至少有一个正样本锚框
    for gt_idx in range(len(gt_boxes)):
        if max_iou_per_gt[gt_idx] >= min_pos_iou:
            # 找到与此gt IoU最大的锚框
            anchor_idx = iou_matrix[:, gt_idx].argmax()
            # 强制设为正样本
            labels[anchor_idx] = 1
            # 更新匹配关系
            gt_idx_per_anchor[anchor_idx] = gt_idx

    # 7. 负样本：IoU < neg_iou_thresh 且不是正样本
    neg_mask = (max_iou_per_anchor < neg_iou_thresh) & (labels != 1)
    labels[neg_mask] = 0

    # === 8. 正负样本采样 (Sampling) ===

    # 正样本采样
    num_fg = int(total_sample_size * pos_fraction)
    fg_indices = torch.where(labels == 1)[0]
    if len(fg_indices) > num_fg:
        # 随机选择丢弃的正样本
        disable_inds = fg_indices[
            torch.randperm(len(fg_indices), device=device)[: len(fg_indices) - num_fg]
        ]
        labels[disable_inds] = -1

    # 负样本采样
    # 剩余配额给负样本
    num_bg = total_sample_size - (labels == 1).sum().item()
    bg_indices = torch.where(labels == 0)[0]
    if len(bg_indices) > num_bg:
        # 随机选择丢弃的负样本
        disable_inds = bg_indices[
            torch.randperm(len(bg_indices), device=device)[: len(bg_indices) - num_bg]
        ]
        labels[disable_inds] = -1

    # 9. 计算正样本的回归目标
    pos_indices = torch.where(labels == 1)[0]
    if len(pos_indices) > 0:
        # 向量化计算回归目标
        matched_gt_boxes = gt_boxes[gt_idx_per_anchor[pos_indices]]
        matched_anchors = anchors[pos_indices]

        ax = matched_anchors[:, 0]
        ay = matched_anchors[:, 1]
        aw = matched_anchors[:, 2] - matched_anchors[:, 0]
        ah = matched_anchors[:, 3] - matched_anchors[:, 1]

        gx = matched_gt_boxes[:, 0]
        gy = matched_gt_boxes[:, 1]
        gw = matched_gt_boxes[:, 2] - matched_gt_boxes[:, 0]
        gh = matched_gt_boxes[:, 3] - matched_gt_boxes[:, 1]

        tx = (gx - ax) / aw
        ty = (gy - ay) / ah
        tw = torch.log(gw / aw)
        th = torch.log(gh / ah)

        reg_targets[pos_indices, 0] = tx
        reg_targets[pos_indices, 1] = ty
        reg_targets[pos_indices, 2] = tw
        reg_targets[pos_indices, 3] = th

    return labels, reg_targets


def assign_targets_to_proposals(
    proposals: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    device: torch.device,
    pos_iou_thresh: float = 0.5,
    neg_iou_thresh_hi: float = 0.5,
    neg_iou_thresh_lo: float = 0.0,
    batch_size_per_image: int = 128,
    positive_fraction: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    为Fast R-CNN分配目标并进行采样 (Sampling)

    Args:
        proposals: [N, 4] (N usually 2000)
        gt_boxes: [M, 4]
        gt_labels: [M] (真实类别, >0)
        device: device
        pos_iou_thresh: 正样本IoU阈值
        neg_iou_thresh_hi: 负样本IoU阈值上限
        neg_iou_thresh_lo: 负样本IoU阈值下限
        batch_size_per_image: 每张图片采样的RoI数量
        positive_fraction: 正样本比例

    Returns:
        sampled_proposals: [batch_size_per_image, 4]
        sampled_labels: [batch_size_per_image] (0为背景)
        sampled_reg_targets: [batch_size_per_image, 4]
    """
    # 1. 计算IoU
    iou_matrix = box_iou(proposals, gt_boxes)  # [N, M]

    # 2. 每个proposal的最佳匹配GT
    max_iou_per_proposal, gt_idx_per_proposal = iou_matrix.max(dim=1)  # [N]

    # 3. 初始化标签 (0: Background)
    labels = torch.zeros(proposals.shape[0], dtype=torch.long, device=device)

    # 4. 正样本: IoU >= 0.5
    pos_mask = max_iou_per_proposal >= pos_iou_thresh
    labels[pos_mask] = gt_labels[gt_idx_per_proposal[pos_mask]]

    # 5. 负样本: lo <= IoU < hi (labels already 0)
    # 忽略那些 IoU < lo 的样本 (设为 -1)
    # neg_mask = (max_iou_per_proposal >= neg_iou_thresh_lo) & (max_iou_per_proposal < neg_iou_thresh_hi)
    # 但通常我们将所有非正样本视为负样本候选，然后在采样时控制

    # 6. 采样
    num_pos = int(batch_size_per_image * positive_fraction)
    pos_indices = torch.where(pos_mask)[0]

    if len(pos_indices) > num_pos:
        # 随机丢弃多余的正样本
        # shuffle and pick num_pos
        keep_pos_indices = pos_indices[
            torch.randperm(len(pos_indices), device=device)[:num_pos]
        ]
    else:
        keep_pos_indices = pos_indices

    # 负样本采样
    num_neg = batch_size_per_image - len(keep_pos_indices)
    neg_mask = (max_iou_per_proposal >= neg_iou_thresh_lo) & (
        max_iou_per_proposal < neg_iou_thresh_hi
    )
    neg_indices = torch.where(neg_mask)[0]

    if len(neg_indices) > num_neg:
        keep_neg_indices = neg_indices[
            torch.randperm(len(neg_indices), device=device)[:num_neg]
        ]
    else:
        # 如果负样本不够，可能会导致总数少于128，这通常是可以接受的
        keep_neg_indices = neg_indices

    # 合并索引
    keep_indices = torch.cat([keep_pos_indices, keep_neg_indices])

    # 提取采样后的数据
    sampled_proposals = proposals[keep_indices]
    sampled_labels = labels[keep_indices]

    # 计算回归目标 (仅针对正样本，负样本为0或忽略)
    # 编码: (gt - proposal) / proposal_size
    matched_gt_boxes = gt_boxes[gt_idx_per_proposal[keep_indices]]
    sampled_reg_targets = encode_boxes(matched_gt_boxes, sampled_proposals)

    return sampled_proposals, sampled_labels, sampled_reg_targets


if __name__ == "__main__":
    # test get_categories_save_to_json_file
    # dataset_path: data/raw/VOCdevkit2007/VOC2007
    dataset_path = Path("data/raw/VOCdevkit2007/VOC2007")
    get_categories_save_to_json_file(dataset_path)
