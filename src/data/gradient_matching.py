"""
梯度匹配数据蒸馏模块
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
    # 如果直接运行，使用相对导入
    from .base_distiller import BaseDistiller

class GradientMatching(BaseDistiller):
    """梯度匹配数据蒸馏"""

    def __init__(self, config: dict, device: torch.device):
        super().__init__(config, device)
        self.synthesis_steps = config.get('synthesis_steps', 1000)
        self.inner_loop = config.get('inner_loop', 1)
        self.lr_syn = config.get('lr_syn', 0.01)
        self.lr_net = config.get('lr_net', 0.01)
        self.init_method = config.get('init_method', 'real')

    def distill(self, train_loader: torch.utils.data.DataLoader,
                num_classes: int,
                teacher_model: Optional[nn.Module] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """执行梯度匹配数据蒸馏"""
        print(f"开始梯度匹配数据蒸馏...")
        print(f"目标: 每类 {self.ipc} 张图像，共 {num_classes * self.ipc} 张")

        # 1. 初始化合成数据（使用基类的方法）
        synthetic_data, synthetic_labels = self._initialize_synthetic_data(
            train_loader, num_classes
        )

        synthetic_data = synthetic_data.to(self.device)
        synthetic_labels = synthetic_labels.to(self.device)
        synthetic_data.requires_grad_(True)

        # 2. 创建学生网络（使用基类的方法）
        student_model = self.create_default_student_model(num_classes).to(self.device)

        # 3. 优化器
        optimizer = optim.Adam([synthetic_data], lr=self.lr_syn)

        # 4. 梯度匹配训练
        print("进行梯度匹配训练...")
        progress_bar = tqdm(range(self.synthesis_steps), desc="梯度匹配")

        best_data = None
        best_loss = float('inf')

        for step in progress_bar:
            # 采样真实数据
            try:
                real_images, real_labels = next(iter(train_loader))
                real_images = real_images.to(self.device)
                real_labels = real_labels.to(self.device)
            except:
                continue

            # 重置学生模型
            student_model.apply(self._reset_weights)

            # 内循环：在合成数据上训练
            for _ in range(self.inner_loop):
                student_model.train()
                syn_output = student_model(synthetic_data)
                syn_loss = F.cross_entropy(syn_output, synthetic_labels)

                syn_grad = torch.autograd.grad(
                    syn_loss, student_model.parameters(),
                    create_graph=True, retain_graph=True
                )

                self._update_model(student_model, syn_grad, self.lr_net)

            # 计算真实数据梯度
            student_model.eval()
            real_output = student_model(real_images)
            real_loss = F.cross_entropy(real_output, real_labels)
            real_grad = torch.autograd.grad(real_loss, student_model.parameters())

            # 重新计算合成数据梯度
            student_model.train()
            syn_output_after = student_model(synthetic_data)
            syn_loss_after = F.cross_entropy(syn_output_after, synthetic_labels)
            syn_grad_after = torch.autograd.grad(
                syn_loss_after, student_model.parameters(),
                create_graph=True
            )

            # 计算梯度匹配损失
            grad_loss = self._compute_gradient_loss(syn_grad_after, real_grad)

            # 优化合成数据
            optimizer.zero_grad()
            grad_loss.backward()
            optimizer.step()

            # 记录最佳结果
            if grad_loss.item() < best_loss:
                best_loss = grad_loss.item()
                best_data = synthetic_data.detach().clone()

            progress_bar.set_postfix({"损失": f"{grad_loss.item():.6f}"})

            # 保存中间结果（使用基类的方法）
            if (step + 1) % 100 == 0:
                self.save_synthetic_images(
                    synthetic_data, synthetic_labels, step + 1,
                    filename="gradient_matching"
                )

        print("数据蒸馏完成!")
        return best_data.cpu(), synthetic_labels.cpu()

    def _reset_weights(self, m):
        """重置模型权重"""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def _update_model(self, model: nn.Module, gradients, lr: float):
        """更新模型"""
        for param, grad in zip(model.parameters(), gradients):
            if grad is not None:
                param.data = param.data - lr * grad

    def _compute_gradient_loss(self, grad1, grad2):
        """计算梯度损失"""
        loss = 0.0
        count = 0
        for g1, g2 in zip(grad1, grad2):
            if g1 is not None and g2 is not None:
                loss += F.mse_loss(g1, g2)
                count += 1
        return loss / max(count, 1)