import torch
import torch.nn as nn


class FasterRcnnHead(nn.Module):
    def __init__(self, input_dim: int, mid_dim: int, output_dim: int) -> None:
        """
        Args:
            input_dim: C * pooled_size * pooled_size (例如 512 * 7 * 7)
            mid_dim: 全连接层隐藏层大小 (通常是 4096 或 1024)
            output_dim: num_classes (包含背景)
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, mid_dim)

        # 最终分叉
        self.cls_fc = nn.Linear(mid_dim, output_dim)
        self.reg_fc = nn.Linear(mid_dim, output_dim * 4)

        self.relu = nn.ReLU()

    def forward(self, pooled_feature_map: torch.Tensor):
        x = pooled_feature_map.flatten(start_dim=1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        cls_logits = self.cls_fc(x)

        reg_logits = self.reg_fc(x)

        return cls_logits, reg_logits
