from typing import Dict, List

import torch
from torch import Tensor

from faster_rcnn.utils.boxes import box_iou


def calculate_map(
    det_boxes: List[Tensor],
    det_labels: List[Tensor],
    det_scores: List[Tensor],
    gt_boxes: List[Tensor],
    gt_labels: List[Tensor],
    iou_threshold: float = 0.5,
    num_classes: int = 20,
    background_class_id: int = 0,
) -> Dict[str, float]:
    """
    Calculate Mean Average Precision (mAP) for object detection.

    Args:
        det_boxes: List of predicted bounding boxes [N, 4] for each image.
        det_labels: List of predicted labels [N] for each image.
        det_scores: List of predicted scores [N] for each image.
        gt_boxes: List of ground truth bounding boxes [M, 4] for each image.
        gt_labels: List of ground truth labels [M] for each image.
        iou_threshold: IoU threshold for matching predictions to ground truth.
        num_classes: Number of classes (excluding background).

    Returns:
        Dictionary containing mAP and AP per class.
    """
    average_precisions = {}

    # Process each class
    for class_idx in range(num_classes):
        # Gather all predictions and ground truths for this class across all images
        class_detections = []
        class_ground_truths = []
        n_ground_truths = 0

        for i in range(len(det_boxes)):
            # Filter predictions for this class
            det_mask = det_labels[i] == class_idx
            if det_mask.sum() > 0:
                class_detections.append(
                    {
                        "boxes": det_boxes[i][det_mask],
                        "scores": det_scores[i][det_mask],
                        "image_idx": i,
                    }
                )

            # Filter ground truths for this class
            gt_mask = gt_labels[i] == class_idx
            n_gt = gt_mask.sum().item()
            if n_gt > 0:
                class_ground_truths.append(
                    {
                        "boxes": gt_boxes[i][gt_mask],
                        "image_idx": i,
                        "used": torch.zeros(n_gt, dtype=torch.bool),
                    }
                )
            n_ground_truths += n_gt

        # If no ground truths for this class, AP is 0 (or undefined, but 0 is standard)
        if n_ground_truths == 0:
            average_precisions[class_idx] = 0.0
            continue

        # If no detections, AP is 0
        if not class_detections:
            average_precisions[class_idx] = 0.0
            continue

        # Flatten detections
        all_det_boxes = torch.cat([d["boxes"] for d in class_detections], dim=0)
        all_det_scores = torch.cat([d["scores"] for d in class_detections], dim=0)
        all_det_img_idxs = []
        for d in class_detections:
            all_det_img_idxs.extend([d["image_idx"]] * len(d["boxes"]))

        # Sort by score descending
        sort_inds = torch.argsort(all_det_scores, descending=True)
        all_det_boxes = all_det_boxes[sort_inds]
        all_det_img_idxs = [all_det_img_idxs[i] for i in sort_inds.tolist()]

        TP = torch.zeros(len(all_det_boxes))
        FP = torch.zeros(len(all_det_boxes))

        # Organize GT by image index for fast lookup
        gt_by_image = {gt["image_idx"]: gt for gt in class_ground_truths}

        for i in range(len(all_det_boxes)):
            img_idx = all_det_img_idxs[i]
            det_box = all_det_boxes[i].unsqueeze(0)

            if img_idx in gt_by_image:
                gt_data = gt_by_image[img_idx]
                gt_box_list = gt_data["boxes"]

                ious = box_iou(det_box, gt_box_list).squeeze(0)
                max_iou, max_idx = torch.max(ious, dim=0)

                if max_iou >= iou_threshold:
                    if not gt_data["used"][max_idx]:
                        TP[i] = 1
                        gt_data["used"][max_idx] = True
                    else:
                        FP[i] = 1
                else:
                    FP[i] = 1
            else:
                FP[i] = 1

        # Calculate Precision and Recall
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = TP_cumsum / n_ground_truths
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)

        # 11-point interpolation (Pascal VOC 2007 style)
        ap = 0.0
        for t in torch.arange(0, 1.1, 0.1):
            if torch.sum(recalls >= t) == 0:
                p = 0
            else:
                p = torch.max(precisions[recalls >= t])
            ap += p / 11.0

        average_precisions[class_idx] = ap.item()

    # Calculate mAP
    valid_aps = [v for k, v in average_precisions.items() if k != background_class_id]
    mAP = sum(valid_aps) / len(valid_aps) if valid_aps else 0.0

    return {"map": mAP, **{f"ap_class_{k}": v for k, v in average_precisions.items()}}
