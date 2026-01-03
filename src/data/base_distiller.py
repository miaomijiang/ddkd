"""
基础蒸馏器类
"""
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional
import numpy as np

class BaseDistiller(ABC):
    """数据蒸馏器基类"""

    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.ipc = config.get('ipc', 50)  # 每类图像数

        # 数据集信息（将在distill时设置）
        self.dataset_info = None
        self.input_shape = None
        self.num_channels = 3  # 默认值

    def set_dataset_info(self, dataset_info: Dict):
        """设置数据集信息"""
        self.dataset_info = dataset_info
        self.input_shape = dataset_info.get('input_shape', (3, 32, 32))
        self.num_channels = self.input_shape[0]

    @abstractmethod
    def distill(self, train_loader: torch.utils.data.DataLoader,
                num_classes: int,
                teacher_model: Optional[nn.Module] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行数据蒸馏（子类必须实现）"""
        pass

    def create_default_student_model(self, num_classes: int) -> nn.Module:
        """创建默认的学生模型"""
        class SimpleCNN(nn.Module):
            def __init__(self, in_channels, num_classes):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(in_channels, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.AdaptiveAvgPool2d((1, 1))
                )
                self.classifier = nn.Linear(64, num_classes)

            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                return self.classifier(x)

        return SimpleCNN(self.num_channels, num_classes)

    def get_normalization_params(self):
        """获取归一化参数"""
        if self.dataset_info and 'mean' in self.dataset_info:
            mean = self.dataset_info['mean']
            std = self.dataset_info['std']
            # 确保参数格式正确
            if isinstance(mean, (int, float)):
                mean = (mean,) * self.num_channels
            if isinstance(std, (int, float)):
                std = (std,) * self.num_channels
            return mean, std
        else:
            # 默认值
            return (0.5,) * self.num_channels, (0.5,) * self.num_channels

    def save_synthetic_images(self, images: torch.Tensor, labels: torch.Tensor,
                              step: int, filename: str = None):
        """通用方法：保存合成图像"""
        import os
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # 创建保存目录
        save_dir = "synthetic_images"
        os.makedirs(save_dir, exist_ok=True)

        # 获取归一化参数
        mean, std = self.get_normalization_params()
        mean = np.array(mean).reshape(-1, 1, 1)
        std = np.array(std).reshape(-1, 1, 1)

        # 可视化设置
        num_classes = len(torch.unique(labels))
        show_classes = min(num_classes, 5)
        show_ipc = min(self.ipc, 5)

        fig, axes = plt.subplots(show_classes, show_ipc,
                                 figsize=(show_ipc * 2, show_classes * 2))

        if show_classes == 1:
            axes = axes.reshape(1, -1)

        for i in range(show_classes):
            class_mask = (labels == i)
            class_images = images[class_mask]

            for j in range(show_ipc):
                if j < len(class_images):
                    img = class_images[j].cpu().detach().numpy()

                    # 反归一化
                    img = img * std + mean
                    img = np.clip(img, 0, 1)

                    # 调整通道顺序
                    if img.shape[0] == 3:  # RGB
                        img = np.transpose(img, (1, 2, 0))
                    elif img.shape[0] == 1:  # 灰度
                        img = img[0]

                    ax = axes[i, j] if show_classes > 1 else axes[j]
                    ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
                    ax.axis('off')

            if show_classes > 1:
                axes[i, 0].set_title(f"Class {i}", fontsize=10)

        method_name = self.__class__.__name__
        plt.suptitle(f"{method_name} - Step {step}", fontsize=12)
        plt.tight_layout()

        # 保存图像
        if filename is None:
            filename = f"{save_dir}/step_{step:04d}.png"
        else:
            filename = f"{save_dir}/{filename}_step_{step:04d}.png"

        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()

    def _initialize_synthetic_data(self, train_loader, num_classes):
        """初始化合成数据"""
        init_method = self.config.get('init_method', 'real')

        if init_method == 'real':
            return self._initialize_from_real(train_loader, num_classes)
        else:
            return self._initialize_random(train_loader, num_classes)

    def _initialize_from_real(self, train_loader, num_classes):
        """从真实数据初始化"""
        class_samples = {i: [] for i in range(num_classes)}

        for images, labels in train_loader:
            for img, label in zip(images, labels):
                label = label.item()
                if len(class_samples[label]) < self.ipc:
                    class_samples[label].append(img.unsqueeze(0))

            if all(len(samples) >= self.ipc for samples in class_samples.values()):
                break

        synthetic_images = []
        synthetic_labels = []

        for class_id in range(num_classes):
            samples = class_samples[class_id]
            if len(samples) >= self.ipc:
                selected = torch.cat(samples[:self.ipc])
            else:
                # 如果某类样本不足，使用随机初始化
                selected = torch.randn(
                    self.ipc, *self.input_shape
                )

            synthetic_images.append(selected)
            synthetic_labels.extend([class_id] * self.ipc)

        return torch.cat(synthetic_images), torch.tensor(synthetic_labels)

    def _initialize_random(self, train_loader, num_classes):
        """随机初始化"""
        synthetic_images = torch.randn(
            num_classes * self.ipc, *self.input_shape
        )
        synthetic_labels = torch.arange(num_classes).repeat_interleave(self.ipc)

        return synthetic_images, synthetic_labels