"""
DC-DSA 数据蒸馏
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from typing import Tuple, Optional

# 导入基类
try:
    from base_distiller import BaseDistiller
except ImportError:
    from .base_distiller import BaseDistiller

class DCDSA(BaseDistiller):
    """DC-DSA数据蒸馏"""

    def __init__(self, config: dict, device: torch.device):
        super().__init__(config, device)
        self.synthesis_steps = config.get('synthesis_steps', 1000)
        self.lr = config.get('lr', 0.1)

    def distill(self, train_loader, num_classes, teacher_model=None, **kwargs):
        """执行DC-DSA数据蒸馏"""
        print(f"开始DC-DSA数据蒸馏...")

        # 初始化合成数据（使用基类的方法）
        synthetic_data, synthetic_labels = self._initialize_synthetic_data(
            train_loader, num_classes
        )

        synthetic_data = synthetic_data.to(self.device)
        synthetic_labels = synthetic_labels.to(self.device)
        synthetic_data.requires_grad_(True)

        # 优化器
        optimizer = optim.Adam([synthetic_data], lr=self.lr)

        # 训练循环
        for epoch in range(self.synthesis_steps):
            total_loss = 0
            batch_count = 0

            for real_images, real_labels in train_loader:
                real_images = real_images.to(self.device)
                real_labels = real_labels.to(self.device)

                # 应用增强
                aug_synthetic = self._apply_augmentation(synthetic_data)

                # 计算对比损失
                loss = self._compute_contrastive_loss(
                    aug_synthetic, synthetic_labels,
                    real_images, real_labels
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            if (epoch + 1) % 100 == 0:
                avg_loss = total_loss / max(batch_count, 1)
                print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
                # 保存中间结果（使用基类的方法）
                self.save_synthetic_images(
                    synthetic_data, synthetic_labels, epoch + 1,
                    filename="dc_dsa"
                )

        print("DC-DSA蒸馏完成!")
        return synthetic_data.detach().cpu(), synthetic_labels.cpu()

    def _apply_augmentation(self, images):
        """应用数据增强"""
        # 简单实现：随机裁剪和翻转
        batch_size = images.size(0)

        # 随机裁剪
        pad_size = 4
        images_padded = F.pad(images, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

        h_idx = torch.randint(0, 9, (batch_size,)).to(self.device)
        w_idx = torch.randint(0, 9, (batch_size,)).to(self.device)

        cropped = torch.stack([
            images_padded[i, :, h_idx[i]:h_idx[i]+images.shape[2], w_idx[i]:w_idx[i]+images.shape[3]]
            for i in range(batch_size)
        ])

        # 随机水平翻转
        flip_mask = torch.rand(batch_size, device=self.device) > 0.5
        cropped[flip_mask] = torch.flip(cropped[flip_mask], dims=[3])

        return cropped

    def _compute_contrastive_loss(self, syn_images, syn_labels, real_images, real_labels):
        """计算对比损失"""
        # 合并所有图像
        all_images = torch.cat([syn_images, real_images], dim=0)
        all_labels = torch.cat([syn_labels, real_labels], dim=0)

        batch_size = all_images.size(0)

        # 计算特征相似度
        images_flat = all_images.view(batch_size, -1)
        images_norm = F.normalize(images_flat, dim=1)
        similarity = torch.mm(images_norm, images_norm.t())

        # 创建标签掩码
        label_matrix = all_labels.unsqueeze(1) == all_labels.unsqueeze(0)

        # 正样本对损失
        pos_mask = label_matrix.float()
        pos_mask.fill_diagonal_(0)
        pos_loss = -torch.log(torch.exp(similarity * pos_mask).sum(dim=1) + 1e-8).mean()

        # 负样本对损失
        neg_mask = (~label_matrix).float()
        neg_loss = torch.log(torch.exp(similarity * neg_mask).sum(dim=1) + 1e-8).mean()

        return pos_loss + neg_loss