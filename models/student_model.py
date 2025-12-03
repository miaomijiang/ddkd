import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet18_Weights
from config.config import *


class StudentModel(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentModel, self).__init__()
        # 加载ResNet18（规范权重参数）
        if PRETRAINED:
            self.base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.base_model = models.resnet18(weights=None)

        # 替换最后一层全连接（适配类别数）
        in_features = self.base_model.fc.in_features  # ResNet18的in_features=512
        self.base_model.fc = nn.Linear(in_features, num_classes)

        # 保存中间特征（layer4输出）
        self.mid_features = None

    def forward(self, x):
        """明确前向传播流程：与教师模型结构对齐"""
        # 1. 前向传播到layer4
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)  # 与教师模型同层级中间特征
        self.mid_features = x

        # 2. avgpool+展平+全连接
        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)

        return x

    def get_mid_features(self):
        return self.mid_features