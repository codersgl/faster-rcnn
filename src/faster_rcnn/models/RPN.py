import math
from typing import Tuple

import torch
import torch.nn as nn
from torchvision.ops import nms

from faster_rcnn.utils.boxes import decode_boxes


class RPN(nn.Module):
    """Region Proposal Network (RPN) for object detection."""

    def __init__(
        self,
        input_channels: int,
        mid_channels: int,
        num_anchors: int,
        im_size: Tuple[int, int],
        min_size: int = 16,
        nms_thresh: float = 0.7,
        base_size: int = 16,
        stride: int = 16,
        scale: float = 1.0,
        scales=[8, 16, 32],
        ratios=[0.5, 1, 1.5],
        num_sample_before_nms: int = 2000,
        num_sample_after_nms: int = 2000,
        device="cpu",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, mid_channels, kernel_size=3, padding=1)

        self.cls_layer = nn.Conv2d(mid_channels, num_anchors * 2, kernel_size=1)
        self.reg_layer = nn.Conv2d(mid_channels, num_anchors * 4, kernel_size=1)

        self.anchor_generator = AnchorGenerator(
            base_size, scales, ratios, stride, device
        )

        self.proposal_generator = ProposalGenerator(
            im_size,
            min_size,
            nms_thresh,
            scale,
            num_sample_before_nms,
            num_sample_after_nms,
        )

        self.relu = nn.ReLU()

    def forward(self, feature):
        batch_size = feature.size(0)

        # output: [batch_size, mid_channels, height, width]
        x = self.relu(self.conv(feature))

        # output: [batch_size, num_anchors * 2, height, width]
        cls_logits = self.cls_layer(x)
        # output: [batch_size, height, width, num_anchors * 2]
        cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous()
        # output: [batch_size, num_anchors * height * width, 2]
        cls_logits = cls_logits.view(batch_size, -1, 2)

        # 添加softmax得到概率分数
        cls_score = torch.softmax(cls_logits, dim=-1)

        # output: [batch_size, num_anchors * 4, height, width]
        reg_logits = self.reg_layer(x)
        # output: [batch_size, height, width, num_anchors * 4]
        reg_logits = reg_logits.permute(0, 2, 3, 1).contiguous()
        # output: [batch_size, num_anchors * height * width, 4]
        reg_logits = reg_logits.view(batch_size, -1, 4)

        anchors = self.anchor_generator(feature)
        proposals = self.proposal_generator(anchors, cls_score, reg_logits)

        return proposals


class AnchorGenerator(nn.Module):
    """Generate anchors for a given feature map size."""

    def __init__(self, base_size, scales, ratios, stride, device: torch.types.Device):
        super().__init__()
        self.base_size = base_size
        self.scales = scales
        self.ratios = ratios
        self.stride = stride
        self.device = device

        self.base_anchor_boxes = self._generate_base_anchor_boxes()

    def forward(self, feature):
        batch_size = feature.size(0)
        height, width = feature.size(2), feature.size(3)

        anchor_boxes = self._generate_all_anchors(self.base_anchor_boxes, width, height)
        anchor_boxes = anchor_boxes.unsqueeze(0).expand(batch_size, -1, -1)

        return anchor_boxes

    def _generate_base_anchor_boxes(self) -> torch.Tensor:
        """Generate the base anchor boxes for feature map

        Return:
            base_anchor_boxes(torch.Tensor): shape[num_scales * num_ratios, 4], [:, x1, y1, x2, y2]
        """
        num_anchors = len(self.scales) * len(self.ratios)
        base_anchor_boxes = torch.zeros(
            (num_anchors, 4), dtype=torch.float32, device=self.device
        )

        cx, cy = (self.base_size - 1) / 2.0

        anchor_idx = 0
        for scale in self.scales:
            for ratio in self.ratios:
                w = self.base_size * scale * math.sqrt(ratio)
                h = self.base_size * scale / math.sqrt(ratio)
                base_anchor_boxes[anchor_idx, 0] = cx - w / 2.0
                base_anchor_boxes[anchor_idx, 1] = cy - h / 2.0
                base_anchor_boxes[anchor_idx, 2] = cx + w / 2.0
                base_anchor_boxes[anchor_idx, 3] = cy + h / 2.0
                anchor_idx += 1

        return base_anchor_boxes

    def _generate_all_anchors(self, base_anchor_boxes, width, height) -> torch.Tensor:
        """Generate all anchors for any position in feature map
        Args:
            base_anchor_boxes(torch.tensor): [num_anchors, 4]
            width(int): The width of feature map
            height(int): The height of feature map
        Return:
            anchors(torch.tensor): [num_positions * num_anchors, 4]
        """
        shift_x = (
            torch.arange(0, width, dtype=torch.float32, device=self.device)
            * self.stride
        )
        shift_y = (
            torch.arange(0, height, dtype=torch.float32, device=self.device)
            * self.stride
        )

        shift_x, shift_y = torch.meshgrid(shift_x, shift_y, indexing="xy")

        shift_x_flat = shift_x.reshape(-1)
        shift_y_flat = shift_y.reshape(-1)

        shifts = torch.stack(
            (shift_x_flat, shift_y_flat, shift_x_flat, shift_y_flat), dim=1
        )

        # [1, num_anchors, 4] + [num_positions, 1, 4] = [num_positions, num_anchors, 4]
        anchors = base_anchor_boxes.unsqueeze(0) + shifts.unsqueeze(1)

        # [num_positions * num_anchors, 4]
        anchors = anchors.reshape(-1, 4)

        return anchors


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
        cls_score: torch.Tensor,
        reg_logits: torch.Tensor,
    ):
        """Get proposal
        Args:
            anchors(torch.Tensor): [batch_size, num_position * num_anchors, 4]
            cls_score(torch.Tensor): [batch_size, num_anchors * height * width, 2]
            reg_logits(torch.Tensor): [batch_size, num_anchors * height * width, 4]
        Return:
            proposals(torch.Tensor): [batch_size, num_sample_after_nms, 4]
        """
        batch_size = anchors.size(0)
        proposals_list = []

        for i in range(batch_size):
            batch_anchors = anchors[i]  # [num_anchors * num_positions, 4]
            batch_reg_logits = reg_logits[i]  # [num_anchors * num_positions, 4]
            batch_cls_score = cls_score[i]  # [num_anchors * num_positions, 2]

            # 1. 解码边界框
            proposal = decode_boxes(
                batch_reg_logits, batch_anchors
            )  # [num_anchors * num_positions, 4]

            # 2. 限制边界框在图像范围内
            im_width, im_height = self.im_size
            proposal[:, [0, 2]] = torch.clamp(
                proposal[:, [0, 2]], min=0, max=im_width - 1
            )
            proposal[:, [1, 3]] = torch.clamp(
                proposal[:, [1, 3]], min=0, max=im_height - 1
            )

            # 3. 移除尺寸太小的边界框
            min_size = self.min_size * self.scale
            proposal_w = proposal[:, 2] - proposal[:, 0] + 1  # 需要+1
            proposal_h = proposal[:, 3] - proposal[:, 1] + 1  # 需要+1

            is_valid = (proposal_w >= min_size) & (proposal_h >= min_size)

            # 只保留有效的proposals和对应的分数
            valid_proposals = proposal[is_valid]
            valid_cls_score = batch_cls_score[is_valid, :]  # 修复维度索引

            # 4. 获取前景分数（第1个通道是前景，第0个是背景）
            if valid_cls_score.size(0) > 0:
                # 获取前景分数
                fg_scores = valid_cls_score[:, 1]
            else:
                fg_scores = torch.tensor(
                    [], dtype=torch.float32, device=valid_proposals.device
                )

            # 5. 按前景分数降序排序
            if len(fg_scores) > 0:
                sorted_indices = torch.argsort(fg_scores, descending=True)

                # 6. 保留前num_sample_before_nms个
                if (
                    self.num_sample_before_nms > 0
                    and len(sorted_indices) > self.num_sample_before_nms
                ):
                    sorted_indices = sorted_indices[: self.num_sample_before_nms]

                sorted_proposals = valid_proposals[sorted_indices]
                sorted_scores = fg_scores[sorted_indices]

                # 7. 应用NMS
                if len(sorted_proposals) > 0:
                    keep = nms(sorted_proposals, sorted_scores, self.nms_thresh)

                    # 8. 如果数量不足则随机抽取填补
                    if len(keep) < self.num_sample_after_nms:
                        num_needed = self.num_sample_after_nms - len(keep)
                        if len(keep) > 0:
                            # 从已保留的框中随机选择
                            random_indices = torch.randint(
                                0, len(keep), (num_needed,), device=keep.device
                            )
                            keep = torch.cat([keep, keep[random_indices]])
                        else:
                            # 如果没有保留任何框，创建空tensor
                            device = sorted_proposals.device
                            keep = torch.tensor([], dtype=torch.long, device=device)

                    # 9. 截取固定数量的边界框
                    if len(keep) > self.num_sample_after_nms:
                        keep = keep[: self.num_sample_after_nms]

                    if len(keep) > 0:
                        batch_proposals = sorted_proposals[keep]
                    else:
                        # 如果没有proposals，创建空的tensor
                        device = sorted_proposals.device
                        batch_proposals = torch.zeros(
                            (0, 4), dtype=torch.float32, device=device
                        )
                else:
                    device = valid_proposals.device
                    batch_proposals = torch.zeros(
                        (0, 4), dtype=torch.float32, device=device
                    )
            else:
                device = valid_proposals.device
                batch_proposals = torch.zeros(
                    (0, 4), dtype=torch.float32, device=device
                )

            # 10. 确保每个batch有固定数量的proposals
            if batch_proposals.size(0) < self.num_sample_after_nms:
                # 填充到固定数量
                num_to_pad = self.num_sample_after_nms - batch_proposals.size(0)
                if batch_proposals.size(0) > 0:
                    # 随机复制现有的proposals
                    pad_indices = torch.randint(
                        0,
                        batch_proposals.size(0),
                        (num_to_pad,),
                        device=batch_proposals.device,
                    )
                    padding = batch_proposals[pad_indices]
                else:
                    # 如果没有proposals，创建零填充
                    padding = torch.zeros(
                        (num_to_pad, 4),
                        dtype=torch.float32,
                        device=batch_proposals.device,
                    )
                batch_proposals = torch.cat([batch_proposals, padding], dim=0)

            proposals_list.append(batch_proposals.unsqueeze(0))

        # 11. 合并所有batch的结果
        proposals = torch.cat(
            proposals_list, dim=0
        )  # [batch_size, num_sample_after_nms, 4]

        return proposals


if __name__ == "__main__":
    ...
