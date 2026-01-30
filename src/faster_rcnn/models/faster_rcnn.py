from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torchvision.ops import nms

from faster_rcnn.models.backbone import vgg16_backbone
from faster_rcnn.models.heads import FasterRcnnHead
from faster_rcnn.models.roi_pooling import RoiPooling
from faster_rcnn.models.rpn import RPN
from faster_rcnn.utils.boxes import decode_boxes


class FasterRCNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_channels: int = 512,  # VGG-16输出通道数
        rpn_mid_channels: int = 512,
        num_anchors: int = 9,
        im_size: Tuple[int, int] = (1000, 600),
        anchor_config: Optional[Dict] = None,
        proposal_config: Optional[Dict] = None,
        roi_pooled_size: Tuple[int, int] = (7, 7),
        roi_spatial_scale: float = 1.0 / 16,
        head_hidden_dim: int = 4096,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.device = device

        # 骨干网络
        self.backbone = vgg16_backbone()

        # RPN
        self.rpn = RPN(
            input_channels=backbone_channels,
            mid_channels=rpn_mid_channels,
            num_anchors=num_anchors,
            im_size=im_size,
            anchor_config=anchor_config,
            device=device,
        )

        # 提案生成器
        if proposal_config is None:
            proposal_config = {
                "min_size": 16,
                "nms_thresh": 0.7,
                "scale": 1.0,
                "num_sample_before_nms": 2000,
                "num_sample_after_nms": 2000,
            }
        self.proposal_generator = ProposalGenerator(im_size=im_size, **proposal_config)

        # ROI池化
        self.roi_pooling = RoiPooling(roi_pooled_size, roi_spatial_scale)

        # 头部网络
        head_input_dim = backbone_channels * roi_pooled_size[0] * roi_pooled_size[1]
        self.head = FasterRcnnHead(
            input_dim=head_input_dim,
            mid_dim=head_hidden_dim,
            output_dim=num_classes,  # 包括背景
        )

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Args:
            images: (batch_size, 3, H, W)
            targets: 可选，训练时用于计算损失

        Returns:
            训练时: 损失字典
            测试时: (预测类别, 预测边界框)
        """
        # 特征提取
        features = self.backbone(images)

        # RPN
        rpn_cls_logits, rpn_reg_logits, anchors = self.rpn(features)

        # 生成提案
        proposals = self.proposal_generator(
            anchors,
            rpn_cls_logits,
            rpn_reg_logits,
            img_size=(images.shape[3], images.shape[2]),
        )

        # ROI池化
        pooled_features = self.roi_pooling(features, proposals)

        # 头部网络
        cls_logits, reg_logits = self.head(pooled_features)

        return cls_logits, reg_logits

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        detections_per_img: int = 100,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Args:
            images: (batch_size, 3, H, W)
            score_thresh: Score threshold
            nms_thresh: NMS threshold
            detections_per_img: Max detections per image

        Returns:
            List of dicts, each with keys 'boxes', 'labels', 'scores'
        """
        self.eval()
        features = self.backbone(images)

        # RPN
        rpn_cls_logits, rpn_reg_logits, anchors = self.rpn(features)
        proposals = self.proposal_generator(
            anchors,
            rpn_cls_logits,
            rpn_reg_logits,
            img_size=(images.shape[3], images.shape[2]),
        )

        # ROI Head
        pooled_features = self.roi_pooling(features, proposals)
        cls_logits, reg_logits = self.head(pooled_features)

        # Post-processing
        probs = torch.softmax(cls_logits, dim=-1)

        # Split back to per-image
        boxes_per_image = [p.shape[0] for p in proposals]
        probs_list = probs.split(boxes_per_image, dim=0)
        reg_logits_list = reg_logits.split(boxes_per_image, dim=0)

        results = []
        num_classes = self.num_classes

        for i, (p, prob, reg) in enumerate(zip(proposals, probs_list, reg_logits_list)):
            # p: [N, 4]
            # prob: [N, num_classes]
            # reg: [N, num_classes * 4]

            # reg: [N, C*4] -> [N, C, 4]
            reg = reg.view(-1, num_classes, 4)

            final_boxes = []
            final_scores = []
            final_labels = []

            for c in range(1, num_classes):  # Skip background (0)
                # Decode boxes for this class
                # p is [N, 4], reg[:, c] is [N, 4]
                boxes_c = decode_boxes(reg[:, c], p)
                scores_c = prob[:, c]

                # Clip boxes
                h, w = images.shape[2], images.shape[3]
                boxes_c[:, [0, 2]] = boxes_c[:, [0, 2]].clamp(min=0, max=w - 1)
                boxes_c[:, [1, 3]] = boxes_c[:, [1, 3]].clamp(min=0, max=h - 1)

                # Filter by score
                keep = scores_c > score_thresh
                boxes_c = boxes_c[keep]
                scores_c = scores_c[keep]

                if boxes_c.numel() == 0:
                    continue

                # NMS
                keep_idx = nms(boxes_c, scores_c, nms_thresh)
                boxes_c = boxes_c[keep_idx]
                scores_c = scores_c[keep_idx]

                labels_c = torch.full_like(scores_c, c, dtype=torch.int64)

                final_boxes.append(boxes_c)
                final_scores.append(scores_c)
                final_labels.append(labels_c)

            if len(final_boxes) > 0:
                final_boxes = torch.cat(final_boxes, dim=0)
                final_scores = torch.cat(final_scores, dim=0)
                final_labels = torch.cat(final_labels, dim=0)

                # Limit detections per image
                if len(final_scores) > detections_per_img:
                    # Sort by score descending
                    sorted_indices = torch.argsort(final_scores, descending=True)
                    sorted_indices = sorted_indices[:detections_per_img]

                    final_boxes = final_boxes[sorted_indices]
                    final_scores = final_scores[sorted_indices]
                    final_labels = final_labels[sorted_indices]
            else:
                final_boxes = torch.empty((0, 4), device=images.device)
                final_scores = torch.empty((0,), device=images.device)
                final_labels = torch.empty(
                    (0,), dtype=torch.int64, device=images.device
                )

            results.append(
                {
                    "boxes": final_boxes,
                    "labels": final_labels,
                    "scores": final_scores,
                }
            )

        return results


class ProposalGenerator(nn.Module):
    def __init__(
        self,
        im_size: Tuple[int, int],
        min_size: int = 16,
        nms_thresh: float = 0.7,
        scale: float = 1.0,
        num_sample_before_nms: int = 2000,
        num_sample_after_nms: int = 2000,
    ):
        super().__init__()
        self.im_size = im_size
        self.scale = scale
        self.min_size = min_size
        self.nms_thresh = nms_thresh
        self.num_sample_before_nms = num_sample_before_nms
        self.num_sample_after_nms = num_sample_after_nms

    def forward(
        self,
        anchors: torch.Tensor,
        cls_logits: torch.Tensor,
        reg_logits: torch.Tensor,
        img_size: Optional[Tuple[int, int]] = None,
    ) -> List[torch.Tensor]:
        """Get proposal
        Args:
            anchors: [batch_size, num_anchors_total, 4]
            cls_logits: [batch_size, num_anchors_total, 2]
            reg_logits: [batch_size, num_anchors_total, 4]
            img_size: Tuple (width, height) used for clipping. If None, use self.im_size.
        Return:
            proposals: [batch_size, num_sample_after_nms, 4]
        """
        # Get probabilities
        cls_score = torch.softmax(cls_logits, dim=-1)

        batch_size = anchors.size(0)
        proposals_list = []

        if img_size is not None:
            im_width, im_height = img_size
        else:
            im_width, im_height = self.im_size

        for i in range(batch_size):
            batch_anchors = anchors[i]
            batch_reg_logits = reg_logits[i]
            batch_cls_score = cls_score[i]

            # 1. Decode boxes
            proposal = decode_boxes(batch_reg_logits, batch_anchors)

            # Filter invalid proposals
            valid_mask = torch.isfinite(proposal).all(dim=1)
            if not valid_mask.all():
                proposal = proposal[valid_mask]
                batch_cls_score = batch_cls_score[valid_mask]

            if proposal.shape[0] == 0:
                proposal = torch.zeros((0, 4), device=anchors.device)
                batch_cls_score = torch.zeros((0, 2), device=anchors.device)

            # 2. Clip boxes
            proposal[:, [0, 2]] = torch.clamp(
                proposal[:, [0, 2]], min=0, max=im_width - 1
            )
            proposal[:, [1, 3]] = torch.clamp(
                proposal[:, [1, 3]], min=0, max=im_height - 1
            )

            # 3. Filter small boxes
            min_size = self.min_size * self.scale
            ws = proposal[:, 2] - proposal[:, 0] + 1
            hs = proposal[:, 3] - proposal[:, 1] + 1
            keep = (ws >= min_size) & (hs >= min_size)

            proposal = proposal[keep]
            scores = batch_cls_score[keep]

            # 4. Get foreground scores
            # scores is [N, 2], we want column 1 (foreground)
            if scores.numel() > 0:
                fg_scores = scores[:, 1]
            else:
                fg_scores = torch.tensor([], device=anchors.device)

            # 5. Sort and pick top N before NMS
            if fg_scores.numel() > 0:
                order = torch.argsort(fg_scores, descending=True)
                if self.num_sample_before_nms > 0:
                    order = order[: self.num_sample_before_nms]

                proposal = proposal[order]
                fg_scores = fg_scores[order]

                # 6. Apply NMS
                keep_idx = nms(proposal, fg_scores, self.nms_thresh)

                # 7. Keep top N after NMS
                if self.num_sample_after_nms > 0:
                    keep_idx = keep_idx[: self.num_sample_after_nms]

                proposal = proposal[keep_idx]

            # 8. Pad if necessary (optional, depending on downstream requirements)
            num_proposals = proposal.size(0)
            if num_proposals < self.num_sample_after_nms:
                num_pad = self.num_sample_after_nms - num_proposals
                if num_proposals > 0:
                    # Random sampling from existing proposals
                    pad_idxs = torch.randint(
                        0, num_proposals, (num_pad,), device=anchors.device
                    )
                    proposal_pad = proposal[pad_idxs]
                else:
                    # Zero padding if no proposals at all
                    proposal_pad = torch.zeros((num_pad, 4), device=anchors.device)

                proposal = torch.cat([proposal, proposal_pad], dim=0)

            proposals_list.append(proposal)

        return proposals_list
