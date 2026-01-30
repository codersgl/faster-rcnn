import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class RPN(nn.Module):
    """Region Proposal Network (RPN) for object detection."""

    def __init__(
        self,
        input_channels: int,
        mid_channels: int,
        num_anchors: int,
        im_size: Tuple[int, int],
        anchor_config: Optional[Dict[str, Any]] = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        # Default configurations if not provided
        if anchor_config is None:
            anchor_config = {
                "base_size": 16,
                "scales": [8, 16, 32],
                "ratios": [0.5, 1, 1.5],
                "stride": 16,
            }

        self.conv = nn.Conv2d(input_channels, mid_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # Classification layer: 2 scores (bg, fg) per anchor
        self.cls_layer = nn.Conv2d(mid_channels, num_anchors * 2, kernel_size=1)
        # Regression layer: 4 coordinates (dx, dy, dw, dh) per anchor
        self.reg_layer = nn.Conv2d(mid_channels, num_anchors * 4, kernel_size=1)

        self.anchor_generator = AnchorGenerator(device=device, **anchor_config)

    def forward(self, feature: torch.Tensor):
        batch_size = feature.size(0)

        # 1. Feature extraction
        # output: [batch_size, mid_channels, height, width]
        x = self.relu(self.conv(feature))

        # 2. Classification logits
        # output: [batch_size, num_anchors * 2, height, width]
        cls_logits = self.cls_layer(x)
        # [batch_size, height, width, num_anchors * 2]
        cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous()
        # [batch_size, num_anchors_total, 2]
        cls_logits = cls_logits.view(batch_size, -1, 2)

        # 3. Regression logits
        # output: [batch_size, num_anchors * 4, height, width]
        reg_logits = self.reg_layer(x)
        # [batch_size, height, width, num_anchors * 4]
        reg_logits = reg_logits.permute(0, 2, 3, 1).contiguous()
        # [batch_size, num_anchors_total, 4]
        reg_logits = reg_logits.view(batch_size, -1, 4)

        # 4. Generate anchors
        # anchors: [batch_size, num_anchors_total, 4]
        anchors = self.anchor_generator(feature)

        return cls_logits, reg_logits, anchors


class AnchorGenerator(nn.Module):
    """Generate anchors for a given feature map size."""

    def __init__(
        self,
        base_size: int,
        scales: List[int],
        ratios: List[float],
        stride: int,
        device: torch.device,
    ):
        super().__init__()
        self.base_size = base_size
        self.scales = scales
        self.ratios = ratios
        self.stride = stride
        self.device = device

        self.base_anchor_boxes = self._generate_base_anchor_boxes()

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch_size = feature.size(0)
        height, width = feature.size(2), feature.size(3)

        anchor_boxes = self._generate_all_anchors(self.base_anchor_boxes, width, height)
        # Expand to batch size: [batch_size, num_anchors, 4]
        anchor_boxes = anchor_boxes.unsqueeze(0).expand(batch_size, -1, -1)

        return anchor_boxes

    def _generate_base_anchor_boxes(self) -> torch.Tensor:
        """Generate the base anchor boxes for feature map"""
        num_anchors = len(self.scales) * len(self.ratios)
        base_anchor_boxes = torch.zeros(
            (num_anchors, 4), dtype=torch.float32, device=self.device
        )

        cx = cy = (self.base_size - 1) / 2.0

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
        """Generate all anchors for any position in feature map"""
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
