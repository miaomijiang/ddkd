"""
混合蒸馏方法 - 简化版
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Tuple, Optional

try:
    from base_distiller import BaseDistiller
except ImportError:
    from .base_distiller import BaseDistiller

class HybridDistillation(BaseDistiller):
    """混合蒸馏：两阶段方法"""

    def __init__(self, config: dict, device: torch.device):
        super().__init__(config, device)

        # 第一阶段配置
        self.gm_steps = config.get('gm_steps', 500)

        # 第二阶段配置
        self.refine_steps = config.get('refine_steps', 200)
        self.refine_lr = config.get('refine_lr', 0.01)

    def distill(self, train_loader, num_classes, teacher_model=None, **kwargs):
        """执行混合蒸馏"""
        print("=== 混合蒸馏开始 ===")

        # 第一阶段：梯度匹配
        print("\n阶段1: 梯度匹配...")
        stage1_data, labels = self._stage1_gradient_matching(
            train_loader, num_classes
        )

        # 第二阶段：精炼
        print("\n阶段2: 精炼...")
        final_data, labels = self._stage2_refine(
            stage1_data, labels, train_loader, num_classes
        )

        print("=== 混合蒸馏完成 ===")
        return final_data, labels

    def _stage1_gradient_matching(self, train_loader, num_classes):
        """第一阶段：梯度匹配"""
        # 使用基类的初始化方法
        synthetic_data, synthetic_labels = self._initialize_synthetic_data(
            train_loader, num_classes
        )

        synthetic_data = synthetic_data.to(self.device)
        synthetic_labels = synthetic_labels.to(self.device)
        synthetic_data.requires_grad_(True)

        # 创建学生模型
        student_model = self.create_default_student_model(num_classes).to(self.device)

        # 优化器
        optimizer = optim.Adam([synthetic_data], lr=0.01)

        # 简化版的梯度匹配
        for step in range(self.gm_steps):
            try:
                real_images, real_labels = next(iter(train_loader))
                real_images = real_images.to(self.device)
                real_labels = real_labels.to(self.device)
            except:
                continue

            # 重置学生模型
            student_model.apply(self._reset_weights)

            # 在合成数据上训练
            student_model.train()
            syn_output = student_model(synthetic_data)
            syn_loss = torch.nn.functional.cross_entropy(syn_output, synthetic_labels)

            syn_grad = torch.autograd.grad(
                syn_loss, student_model.parameters(),
                create_graph=True
            )

            # 更新学生模型
            for param, grad in zip(student_model.parameters(), syn_grad):
                if grad is not None:
                    param.data = param.data - 0.01 * grad

            # 计算真实数据梯度
            student_model.eval()
            real_output = student_model(real_images)
            real_loss = torch.nn.functional.cross_entropy(real_output, real_labels)
            real_grad = torch.autograd.grad(real_loss, student_model.parameters())

            # 重新计算合成数据梯度
            student_model.train()
            syn_output_after = student_model(synthetic_data)
            syn_loss_after = torch.nn.functional.cross_entropy(syn_output_after, synthetic_labels)
            syn_grad_after = torch.autograd.grad(
                syn_loss_after, student_model.parameters(),
                create_graph=True
            )

            # 梯度匹配损失
            grad_loss = self._compute_gradient_loss(syn_grad_after, real_grad)

            # 优化
            optimizer.zero_grad()
            grad_loss.backward()
            optimizer.step()

            if (step + 1) % 100 == 0:
                print(f"阶段1 步 {step+1}, 损失: {grad_loss.item():.6f}")

        return synthetic_data.detach().cpu(), synthetic_labels.cpu()

    def _stage2_refine(self, init_data, labels, train_loader, num_classes):
        """第二阶段：精炼"""
        synthetic_data = init_data.clone().to(self.device).requires_grad_(True)
        synthetic_labels = labels.to(self.device)

        # 优化器
        optimizer = optim.Adam([synthetic_data], lr=self.refine_lr)

        # 精炼过程
        for step in range(self.refine_steps):
            try:
                real_images, real_labels = next(iter(train_loader))
                real_images = real_images.to(self.device)
                real_labels = real_labels.to(self.device)
            except:
                continue

            # 应用简单增强
            aug_synthetic = self._apply_augmentation(synthetic_data)

            # 计算特征损失
            loss = self._compute_feature_loss(
                aug_synthetic, real_images
            )

            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (step + 1) % 50 == 0:
                print(f"阶段2 步 {step+1}, 损失: {loss.item():.6f}")

        return synthetic_data.detach().cpu(), synthetic_labels.cpu()

    def _reset_weights(self, m):
        """重置模型权重"""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def _compute_gradient_loss(self, grad1, grad2):
        """计算梯度损失"""
        loss = 0.0
        count = 0
        for g1, g2 in zip(grad1, grad2):
            if g1 is not None and g2 is not None:
                loss += torch.nn.functional.mse_loss(g1, g2)
                count += 1
        return loss / max(count, 1)

    def _apply_augmentation(self, images):
        """应用数据增强"""
        batch_size = images.size(0)

        # 随机裁剪
        pad_size = 4
        images_padded = torch.nn.functional.pad(images, (pad_size, pad_size, pad_size, pad_size), mode='reflect')

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

    def _compute_feature_loss(self, syn_images, real_images):
        """计算特征损失"""
        # 简单的像素级损失
        return torch.nn.functional.mse_loss(syn_images.mean(dim=0), real_images.mean(dim=0))