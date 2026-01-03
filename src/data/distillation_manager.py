"""
数据蒸馏管理器
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

class DistillationManager:
    """数据蒸馏管理器"""

    def __init__(self, config: Dict, device: torch.device):
        self.config = config
        self.device = device
        self.method = config.get('method', 'gradient_matching')
        self.distiller = self._create_distiller()

    def _create_distiller(self):
        """创建蒸馏器实例"""
        method = self.method

        if method == 'gradient_matching':
            try:
                from gradient_matching import GradientMatching
                return GradientMatching(self.config, self.device)
            except ImportError:
                from .gradient_matching import GradientMatching
                return GradientMatching(self.config, self.device)
        elif method == 'dc_dsa':
            try:
                from dc_dsa import DCDSA
                return DCDSA(self.config, self.device)
            except ImportError:
                from .dc_dsa import DCDSA
                return DCDSA(self.config, self.device)
        elif method == 'hybrid':
            try:
                from hybrid_distillation import HybridDistillation
                return HybridDistillation(self.config, self.device)
            except ImportError:
                from .hybrid_distillation import HybridDistillation
                return HybridDistillation(self.config, self.device)
        else:
            raise ValueError(f"不支持的蒸馏方法: {method}")

    def distill(self, dataset_loader, train_loader, teacher_model=None):
        """执行数据蒸馏"""
        # 获取数据集信息
        sample, _ = next(iter(train_loader))
        input_shape = sample.shape[1:]  # (C, H, W)

        dataset_info = {
            'name': dataset_loader.dataset_name,
            'num_classes': dataset_loader.get_num_classes(),
            'num_channels': dataset_loader.num_channels,
            'mean': dataset_loader.mean,
            'std': dataset_loader.std,
            'image_size': dataset_loader.image_size,
            'input_shape': input_shape,
        }

        # 设置蒸馏器数据集信息
        self.distiller.set_dataset_info(dataset_info)

        # 获取类别数
        num_classes = dataset_loader.get_num_classes()

        print(f"开始{self.method}数据蒸馏...")
        print(f"数据集: {dataset_info['name']}")
        print(f"类别数: {num_classes}")
        print(f"输入形状: {input_shape}")

        # 执行蒸馏
        return self.distiller.distill(train_loader, num_classes, teacher_model)