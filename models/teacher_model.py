import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights
from config.config import *


class TeacherModel(nn.Module):
    def __init__(self, num_classes=10):
        super(TeacherModel, self).__init__()
        # 加载ResNet50（规范权重参数）
        if PRETRAINED:
            self.base_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.base_model = models.resnet50(weights=None)

        # 替换最后一层全连接（适配数据集类别数）
        in_features = self.base_model.fc.in_features  # ResNet50的in_features=2048
        self.base_model.fc = nn.Linear(in_features, num_classes)

        # 保存中间特征（layer4输出）
        self.mid_features = None

    def forward(self, x):
        """明确前向传播流程：确保avgpool和fc层正确执行"""
        # 1. 前向传播到layer4（获取中间特征）
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)  # 最后一个卷积层
        self.mid_features = x  # 记录中间特征（用于梯度匹配）

        # 2. 关键：执行avgpool（压缩特征图为1x1）
        x = self.base_model.avgpool(x)
        # 3. 展平特征（2048,1,1 → 2048）
        x = torch.flatten(x, 1)
        # 4. 全连接层输出logits
        x = self.base_model.fc(x)

        return x  # 返回最终logits

    def get_mid_features(self):
        return self.mid_features