import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary
from torchvision.models import VGG16_Weights


def built_backbone():
    """
    层选择：VGG16 的 13 个卷积层为共享部分，需剥离顶部全连接层（分类头），仅保留特征提取部分。
    微调策略：论文中对 VGG16 仅微调 conv3_1 及以上层（节省内存），加载后可按此逻辑设置可训练参数。
    """
    vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    # 使用 Sequential 构造函数创建切片以避免类型检查器错误
    shared_conv_layers = nn.Sequential(*list(vgg16.features.children())[:30])

    # 根据论文，冻结前6层（conv1_1到pool2），从第7层（conv3_1）开始微调
    for param in shared_conv_layers[:6].parameters():
        param.requires_grad = False

    return shared_conv_layers


if __name__ == "__main__":
    backbone = built_backbone()
    x = torch.randn(1, 3, 224, 224)
    summary(backbone, input_size=(1, 3, 224, 224))