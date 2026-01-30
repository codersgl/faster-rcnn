from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision.ops import RoIPool


class RoiPooling(nn.Module):
    def __init__(
        self, pooled_size: Tuple[int, int] = (7, 7), spatial_scale: float = 1.0 / 16
    ) -> None:
        super().__init__()
        self.roi_pool = RoIPool(pooled_size, spatial_scale)

    def forward(self, feature_map: torch.Tensor, proposals: List[torch.Tensor]):
        """
        Args:
            feature_map: (Batch, C, H, W)
            proposals: List of Tensors, length = Batch Size.
                       Each Tensor is (N_i, 4) representing [x1, y1, x2, y2]
        """
        return self.roi_pool(feature_map, proposals)
