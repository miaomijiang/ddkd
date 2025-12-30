"""
模型工厂 - 创建各种模型
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional
import warnings

class ModelFactory:
    """模型工厂"""

    @staticmethod
    def create_model(model_name: str, num_classes: int,
                    pretrained: bool = False, device: Optional[torch.device] = None) -> nn.Module:
        """创建模型，可选择直接移动到设备"""
        # 忽略torchvision的警告
        warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")

        if model_name == "resnet18":
            try:
                # 新版本API（torchvision >= 0.13）
                from torchvision.models import ResNet18_Weights
                weights = ResNet18_Weights.DEFAULT if pretrained else None
                model = models.resnet18(weights=weights)
            except ImportError:
                # 回退到旧API
                model = models.resnet18(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == "resnet34":
            try:
                from torchvision.models import ResNet34_Weights
                weights = ResNet34_Weights.DEFAULT if pretrained else None
                model = models.resnet34(weights=weights)
            except ImportError:
                model = models.resnet34(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == "mobilenet_v2":
            try:
                from torchvision.models import MobileNet_V2_Weights
                weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
                model = models.mobilenet_v2(weights=weights)
            except ImportError:
                model = models.mobilenet_v2(pretrained=pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == "simple_cnn":
            model = SimpleCNN(num_classes)
        else:
            raise ValueError(f"未知模型: {model_name}")

        # 如果指定了设备，直接移动到设备
        if device is not None:
            model = model.to(device)

        return model

    @staticmethod
    def get_model_size(model: nn.Module) -> dict:
        """获取模型大小信息"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        size_mb = param_size / 1024**2

        return {
            'parameters': sum(p.numel() for p in model.parameters()),
            'size_mb': size_mb
        }

class SimpleCNN(nn.Module):
    """简单的CNN模型"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(nn.functional.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x